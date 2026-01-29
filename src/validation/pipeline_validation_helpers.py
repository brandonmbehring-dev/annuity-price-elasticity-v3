"""
Pipeline Validation Helpers - Easy Integration of Production Validators

This module provides convenient wrapper functions for integrating production
validators into data pipeline operations. It follows CODING_STANDARDS.md
fail-fast principles with business context in error messages.

Usage Example:
    from src.validation.pipeline_validation_helpers import validate_extraction_output

    # After data extraction
    df = load_data_from_s3(...)
    df_validated = validate_extraction_output(
        df=df,
        stage_name="sales_data_extraction",
        date_column="date"
    )

Key Features:
- Fail-fast error handling with business context
- Allows data increases (new rows, newer dates)
- Fails on data decreases (fewer rows than previous run)
- Fails on schema violations (missing columns, type changes)
- Fails on unexpected null rate increases

Requirements per User Decision:
- ✅ Allow: Day-to-day data changes (WINK rates, sales)
- ✅ Allow: Data increases (more rows, newer dates)
- ❌ Fail: Decreasing datasets (fewer rows than previous)
- ❌ Fail: Schema violations (missing columns, wrong types)
- ❌ Fail: Unexpected null rate increases (>threshold)

Error Message Pattern:
    CRITICAL: [WHAT happened].
    Stage: {stage}.
    Business impact: [WHY it matters].
    Required action: [WHAT to check].
    Expected: [Normal behavior].
"""

from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path

from src.validation.production_validators import (
    run_production_validation_checkpoint,
    ValidationConfig,
    ValidationResult,
    GrowthConfig
)


# =============================================================================
# Configuration Templates
# =============================================================================

def _get_default_growth_config(allow_shrinkage: bool = False) -> GrowthConfig:
    """Get default growth configuration for data validation.

    Parameters
    ----------
    allow_shrinkage : bool
        If False (default), fail on data shrinkage. If True, only warn.

    Returns
    -------
    GrowthConfig
        Growth validation configuration
    """
    return GrowthConfig({
        'min_growth_pct': -5.0 if allow_shrinkage else 0.0,  # Allow 5% shrinkage if enabled
        'max_growth_pct': 50.0,  # Warn if growth >50%
        'warn_on_shrinkage': True,
        'warn_on_high_growth': True
    })


def _create_validation_config(
    stage_name: str,
    version: int = 6,
    strict_schema: bool = True,
    allow_shrinkage: bool = False,
    critical_columns: Optional[list] = None
) -> ValidationConfig:
    """Create validation configuration for a pipeline stage.

    Parameters
    ----------
    stage_name : str
        Name of the pipeline stage (e.g., "sales_extraction")
    version : int
        Pipeline version number (default: 6)
    strict_schema : bool
        If True, fail on schema changes. If False, warn only.
    allow_shrinkage : bool
        If False (default), fail on data shrinkage
    critical_columns : list, optional
        List of columns that must exist. None = all columns critical.

    Returns
    -------
    ValidationConfig
        Complete validation configuration
    """
    project_root = str(Path.cwd())

    return ValidationConfig({
        'checkpoint_name': stage_name,
        'version': version,
        'project_root': project_root,
        'strict_schema': strict_schema,
        'growth_config': _get_default_growth_config(allow_shrinkage),
        'critical_columns': critical_columns
    })


# =============================================================================
# High-Level Validation Functions
# =============================================================================

def validate_extraction_output(
    df: pd.DataFrame,
    stage_name: str,
    date_column: Optional[str] = "date",
    critical_columns: Optional[list] = None,
    allow_shrinkage: bool = False
) -> pd.DataFrame:
    """Validate data extraction output with fail-fast error handling.

    This function wraps run_production_validation_checkpoint with fail-fast
    behavior and business context per CODING_STANDARDS.md requirements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from data extraction to validate
    stage_name : str
        Name of extraction stage for tracking (e.g., "sales_extraction")
    date_column : str, optional
        Name of date column for time range validation (default: "date")
    critical_columns : list, optional
        Columns that must exist. None = all columns critical.
    allow_shrinkage : bool
        If False (default), fail if row count decreased from previous run

    Returns
    -------
    pd.DataFrame
        The same DataFrame (unmodified) if validation passes

    Raises
    ------
    ValueError
        If validation fails with business context per fail-fast requirement

    Examples
    --------
    >>> df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=100),
    ...                    'sales': range(100)})
    >>> df_validated = validate_extraction_output(df, "sales_extraction")
    >>> assert len(df_validated) == 100  # Validation passed
    """
    config = _create_validation_config(
        stage_name=stage_name,
        strict_schema=True,
        allow_shrinkage=allow_shrinkage,
        critical_columns=critical_columns
    )

    result = run_production_validation_checkpoint(
        df=df,
        config=config,
        date_column=date_column
    )

    # Fail-fast with business context if validation failed
    if result.status == 'FAILED':
        _raise_validation_error(result, stage_name)

    # Warn about non-critical issues but continue
    if result.warnings:
        for warning in result.warnings:
            print(f"WARNING [{stage_name}]: {warning}")

    return df


def validate_preprocessing_output(
    df: pd.DataFrame,
    stage_name: str,
    date_column: Optional[str] = "date",
    allow_shrinkage: bool = True  # Preprocessing may filter rows
) -> pd.DataFrame:
    """Validate preprocessing output with flexible shrinkage rules.

    Preprocessing steps like filtering may legitimately reduce row counts,
    so shrinkage warnings are enabled but don't cause failures by default.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after preprocessing to validate
    stage_name : str
        Name of preprocessing stage (e.g., "product_filtering")
    date_column : str, optional
        Name of date column for validation
    allow_shrinkage : bool
        If True (default for preprocessing), allow row count decreases

    Returns
    -------
    pd.DataFrame
        The same DataFrame (unmodified) if validation passes

    Raises
    ------
    ValueError
        If critical validation fails (schema violations, empty dataset)
    """
    config = _create_validation_config(
        stage_name=stage_name,
        strict_schema=True,
        allow_shrinkage=allow_shrinkage
    )

    result = run_production_validation_checkpoint(
        df=df,
        config=config,
        date_column=date_column
    )

    if result.status == 'FAILED':
        _raise_validation_error(result, stage_name)

    if result.warnings:
        for warning in result.warnings:
            print(f"WARNING [{stage_name}]: {warning}")

    return df


# =============================================================================
# Error Handling with Business Context
# =============================================================================

def _raise_validation_error(result: ValidationResult, stage_name: str) -> None:
    """Raise validation error with business context per CODING_STANDARDS.md.

    Error message pattern:
        CRITICAL: [WHAT happened].
        Stage: {stage}.
        Business impact: [WHY it matters].
        Required action: [WHAT to check].
        Expected: [Normal behavior].

    Parameters
    ----------
    result : ValidationResult
        Validation result with issues
    stage_name : str
        Pipeline stage name

    Raises
    ------
    ValueError
        With detailed business context
    """
    issues_summary = "; ".join(result.issues)

    # Determine business impact based on issue type
    if "schema" in issues_summary.lower():
        business_impact = "Data structure changed - downstream operations will fail"
        required_action = f"Check {stage_name} source data schema. Verify column names and types."
        expected = "Schema should remain stable between runs unless intentionally changed."
    elif "decreased" in issues_summary.lower() or "shrinkage" in issues_summary.lower():
        business_impact = "Data loss detected - analysis will be incomplete"
        required_action = f"Check {stage_name} data source. Verify no data was dropped unexpectedly."
        expected = "Row counts should increase or stay constant as new data arrives."
    elif "empty" in issues_summary.lower():
        business_impact = "No data available - cannot proceed with analysis"
        required_action = f"Check {stage_name} data source connectivity and filters."
        expected = "Dataset should contain rows after extraction/preprocessing."
    else:
        business_impact = "Data quality issue detected"
        required_action = f"Review {stage_name} validation logs for details."
        expected = "Data should pass all quality checks."

    error_message = (
        f"CRITICAL: Data validation failed for {stage_name}. {issues_summary}\n"
        f"Stage: {stage_name}\n"
        f"Business impact: {business_impact}\n"
        f"Required action: {required_action}\n"
        f"Expected: {expected}\n"
        f"\nValidation details:\n"
        f"  Current metrics: {result.current_metrics['record_count']} rows, "
        f"{result.current_metrics['column_count']} columns\n"
    )

    if result.previous_metrics:
        error_message += (
            f"  Previous metrics: {result.previous_metrics['record_count']} rows, "
            f"{result.previous_metrics['column_count']} columns\n"
        )

    raise ValueError(error_message)
