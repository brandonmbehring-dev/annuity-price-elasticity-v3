"""
Production validation checkpoints for data pipeline notebooks.

This module provides reusable validation functions for production data quality
checkpoints in the RILA Price Elasticity Analysis pipeline. It consolidates
repeated validation patterns from notebook cells into testable, maintainable
functions.

Key Features:
- Schema stability validation (strict)
- Data growth pattern validation (flexible warnings)
- Business rule validation (configurable)
- Date range progression checks
- Metadata persistence for run-to-run comparison

Usage Example:
    from src.validation.production_validators import (
        run_production_validation_checkpoint,
        ValidationConfig
    )

    config: ValidationConfig = {
        'checkpoint_name': 'product_filtered',
        'version': 6,
        'project_root': '/path/to/project',
        'strict_schema': True,
        'growth_config': {
            'min_growth_pct': 0.0,
            'max_growth_pct': 20.0,
            'warn_on_shrinkage': True,
            'warn_on_high_growth': True
        },
        'critical_columns': None
    }

    result = run_production_validation_checkpoint(df=my_dataframe, config=config)

Follows CODING_STANDARDS.md:
- Functions are 10-50 lines (atomic operations)
- Complete type hints for all parameters and returns
- Fail-fast error handling with business context
- Zero regression policy (maintains exact validation behavior)
"""

from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


# =============================================================================
# Type Definitions
# =============================================================================

class ValidationMetrics(TypedDict):
    """Metrics collected during a validation run.

    Attributes:
        run_date: ISO format timestamp of validation run
        record_count: Number of records in DataFrame
        column_count: Number of columns in DataFrame
        column_names: List of all column names
        min_date: Minimum date (if date column provided)
        max_date: Maximum date (if date column provided)
        date_range_days: Days between min and max date
        aggregation_ratio: Ratio for aggregation validation (optional)
    """
    run_date: str
    record_count: int
    column_count: int
    column_names: List[str]
    min_date: Optional[str]
    max_date: Optional[str]
    date_range_days: Optional[int]
    aggregation_ratio: Optional[float]


class GrowthConfig(TypedDict):
    """Configuration for growth pattern validation.

    Attributes:
        min_growth_pct: Minimum acceptable growth percentage
        max_growth_pct: Maximum acceptable growth percentage
        warn_on_shrinkage: Whether to warn on data shrinkage
        warn_on_high_growth: Whether to warn on unusually high growth
    """
    min_growth_pct: float
    max_growth_pct: float
    warn_on_shrinkage: bool
    warn_on_high_growth: bool


class ValidationConfig(TypedDict):
    """Complete configuration for a production validation checkpoint.

    Attributes:
        checkpoint_name: Name of validation checkpoint (e.g., 'product_filtered')
        version: Pipeline version number
        project_root: Root directory of project
        strict_schema: Whether to fail on schema changes (True) or warn (False)
        growth_config: Configuration for growth validation
        critical_columns: List of columns that must exist (None = all columns)
    """
    checkpoint_name: str
    version: int
    project_root: str
    strict_schema: bool
    growth_config: GrowthConfig
    critical_columns: Optional[List[str]]


@dataclass
class ValidationResult:
    """Results from a validation checkpoint.

    Attributes:
        checkpoint_name: Name of checkpoint that was validated
        status: 'PASSED' | 'FAILED' | 'BASELINE' | 'WARNINGS'
        issues: List of critical issues that cause failure
        warnings: List of non-critical warnings
        current_metrics: Metrics from current run
        previous_metrics: Metrics from previous run (None if first run)
        metadata_path: Path where metadata was saved
    """
    checkpoint_name: str
    status: str
    issues: List[str]
    warnings: List[str]
    current_metrics: ValidationMetrics
    previous_metrics: Optional[ValidationMetrics]
    metadata_path: Path


# =============================================================================
# Metadata Operations
# =============================================================================

def load_previous_validation_metadata(
    checkpoint_name: str,
    version: int,
    project_root: str
) -> Optional[ValidationMetrics]:
    """Load metadata from previous validation run.

    Args:
        checkpoint_name: Name of validation checkpoint
        version: Pipeline version number
        project_root: Root directory of project

    Returns:
        ValidationMetrics from previous run, or None if no previous run exists
    """
    metadata_path = Path(project_root) / f"outputs/metadata/{checkpoint_name}_v{version}_metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r') as f:
        return json.load(f)


def save_validation_metadata(
    metrics: ValidationMetrics,
    checkpoint_name: str,
    version: int,
    project_root: str
) -> Path:
    """Save current validation metrics to JSON file.

    Args:
        metrics: Validation metrics to save
        checkpoint_name: Name of validation checkpoint
        version: Pipeline version number
        project_root: Root directory of project

    Returns:
        Path to saved metadata file
    """
    metadata_path = Path(project_root) / f"outputs/metadata/{checkpoint_name}_v{version}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metadata_path


# =============================================================================
# Metrics Extraction
# =============================================================================

def extract_validation_metrics(
    df: pd.DataFrame,
    date_column: Optional[str] = None
) -> ValidationMetrics:
    """Extract all validation metrics from DataFrame.

    Args:
        df: DataFrame to extract metrics from
        date_column: Optional name of date column for date range metrics

    Returns:
        ValidationMetrics with all extracted values
    """
    metrics: ValidationMetrics = {
        'run_date': datetime.now().isoformat(),
        'record_count': len(df),
        'column_count': len(df.columns),
        'column_names': list(df.columns),
        'min_date': None,
        'max_date': None,
        'date_range_days': None,
        'aggregation_ratio': None
    }

    # Extract date range if date column provided
    if date_column and date_column in df.columns:
        min_date = df[date_column].min()
        max_date = df[date_column].max()
        metrics['min_date'] = str(min_date)
        metrics['max_date'] = str(max_date)
        metrics['date_range_days'] = (max_date - min_date).days

    return metrics


# =============================================================================
# Schema Validation
# =============================================================================

def validate_schema_stability(
    current_metrics: ValidationMetrics,
    previous_metrics: Optional[ValidationMetrics],
    strict: bool = True,
    critical_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """Validate schema hasn't changed between runs.

    Args:
        current_metrics: Metrics from current run
        previous_metrics: Metrics from previous run (None for first run)
        strict: If True, raise ValueError on schema mismatch. If False, return warnings
        critical_columns: Optional list of columns that must exist

    Returns:
        Tuple of (is_valid, list of issues/warnings)

    Raises:
        ValueError: If strict=True and schema mismatch detected, or if critical columns missing
    """
    issues: List[str] = []
    curr_cols = set(current_metrics['column_names'])

    # Check critical columns ALWAYS (even on first run)
    if critical_columns:
        missing_critical = set(critical_columns) - curr_cols
        if missing_critical:
            raise ValueError(
                f"[CRITICAL]: Missing critical columns: {missing_critical}. "
                "Business impact: Required features unavailable for analysis. "
                f"Required action: Verify upstream pipeline produced {critical_columns}."
            )

    # No schema comparison validation on first run
    if previous_metrics is None:
        return (True, [])

    # Check for schema changes
    prev_cols = set(previous_metrics['column_names'])

    missing_cols = prev_cols - curr_cols
    added_cols = curr_cols - prev_cols

    if missing_cols or added_cols:
        msg = f"Schema mismatch: Missing columns: {missing_cols}, Added columns: {added_cols}"
        if strict:
            raise ValueError(
                f"[CRITICAL]: {msg}. "
                "Business impact: Data structure changed, downstream pipelines may fail. "
                "Required action: Verify schema changes are intentional."
            )
        else:
            issues.append(msg)

    return (len(issues) == 0, issues)


# =============================================================================
# Growth Pattern Validation
# =============================================================================

def validate_growth_patterns(
    current_metrics: ValidationMetrics,
    previous_metrics: Optional[ValidationMetrics],
    config: GrowthConfig
) -> Tuple[bool, List[str], Optional[float]]:
    """Validate data growth patterns between runs.

    Args:
        current_metrics: Metrics from current run
        previous_metrics: Metrics from previous run (None for first run)
        config: Growth validation configuration

    Returns:
        Tuple of (is_valid, list of warnings, growth_percentage)
    """
    warnings: List[str] = []

    # No validation on first run
    if previous_metrics is None:
        return (True, [], None)

    prev_count = previous_metrics['record_count']
    curr_count = current_metrics['record_count']
    growth_pct = ((curr_count - prev_count) / prev_count) * 100

    # Check for shrinkage
    if growth_pct < config['min_growth_pct'] and config['warn_on_shrinkage']:
        warnings.append(
            f"Data shrinkage: {prev_count:,} → {curr_count:,} ({growth_pct:+.1f}%). "
            f"Expected minimum: {config['min_growth_pct']:.1f}%"
        )

    # Check for excessive growth
    if growth_pct > config['max_growth_pct'] and config['warn_on_high_growth']:
        warnings.append(
            f"Unusual growth: {prev_count:,} → {curr_count:,} ({growth_pct:+.1f}%). "
            f"Expected maximum: {config['max_growth_pct']:.1f}%"
        )

    return (True, warnings, growth_pct)


# =============================================================================
# Date Range Validation
# =============================================================================

def validate_date_range_progression(
    current_metrics: ValidationMetrics,
    previous_metrics: Optional[ValidationMetrics],
    max_start_shift_days: int = 30
) -> Tuple[bool, List[str]]:
    """Validate date range progression (historical preservation, current advancement).

    Args:
        current_metrics: Metrics from current run
        previous_metrics: Metrics from previous run (None for first run)
        max_start_shift_days: Maximum allowed shift in start date

    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings: List[str] = []

    # No validation on first run or if no date columns
    if previous_metrics is None or current_metrics['min_date'] is None:
        return (True, [])

    # Parse dates
    prev_min = pd.Timestamp(previous_metrics['min_date'])
    prev_max = pd.Timestamp(previous_metrics['max_date'])
    curr_min = pd.Timestamp(current_metrics['min_date'])
    curr_max = pd.Timestamp(current_metrics['max_date'])

    # Check start date stability (historical data preservation)
    start_shift_days = abs((curr_min - prev_min).days)
    if start_shift_days > max_start_shift_days:
        warnings.append(
            f"Start date shifted significantly: {prev_min} → {curr_min} ({start_shift_days} days). "
            "Historical data may be missing."
        )

    # Check end date progression (data is current or newer)
    if curr_max < prev_max:
        warnings.append(
            f"End date regression: {prev_max} → {curr_max}. Data appears outdated."
        )

    return (True, warnings)


# =============================================================================
# Business Rules Validation
# =============================================================================

def _check_positive_premiums(
    df: pd.DataFrame, rule: Dict[str, Any], raise_on_error: bool = True
) -> Optional[str]:
    """Check for negative premium amounts exceeding threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    rule : Dict[str, Any]
        Rule config with 'column' and 'threshold' keys
    raise_on_error : bool, default=True
        If True, raise exception on failure. If False, return error string.

    Returns
    -------
    Optional[str]
        Violation message if failed and raise_on_error=False, None if passed

    Raises
    ------
    DataValidationError
        If validation fails and raise_on_error=True
    """
    from src.core.exceptions import DataValidationError

    col = rule['column']
    threshold = rule['threshold']
    negative_count = (df[col] < 0).sum()
    if negative_count > threshold:
        error_msg = f"{negative_count} negative premium amounts exceeds threshold ({threshold})"
        if raise_on_error:
            raise DataValidationError(
                error_msg,
                validation_type="positive_premiums",
                business_impact="Revenue calculations will be incorrect",
                required_action="Investigate data quality issues in source system"
            )
        return (
            f"CRITICAL: {error_msg}. "
            "Business impact: Revenue calculations will be incorrect. "
            "Required action: Investigate data quality issues in source system."
        )
    return None


def _check_date_consistency(
    df: pd.DataFrame, rule: Dict[str, Any], raise_on_error: bool = True
) -> Optional[str]:
    """Check that application dates precede contract dates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    rule : Dict[str, Any]
        Rule config with 'app_col' and 'contract_col' keys
    raise_on_error : bool, default=True
        If True, raise exception on failure. If False, return error string.

    Returns
    -------
    Optional[str]
        Violation message if failed and raise_on_error=False, None if passed

    Raises
    ------
    DataValidationError
        If validation fails and raise_on_error=True
    """
    from src.core.exceptions import DataValidationError

    app_col = rule['app_col']
    contract_col = rule['contract_col']
    inconsistencies = (df[app_col] > df[contract_col]).sum()
    if inconsistencies > 0:
        error_msg = f"Found {inconsistencies} records where application date after contract date"
        if raise_on_error:
            raise DataValidationError(
                error_msg,
                validation_type="date_consistency",
                business_impact="Temporal logic violated, may cause incorrect analysis",
                required_action="Review source data date fields"
            )
        return error_msg
    return None


def _check_valid_buffer_rates(
    df: pd.DataFrame, rule: Dict[str, Any], raise_on_error: bool = True
) -> Optional[str]:
    """Check that buffer rates are within valid values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    rule : Dict[str, Any]
        Rule config with 'column' and 'valid_values' keys
    raise_on_error : bool, default=True
        If True, raise exception on failure. If False, return error string.

    Returns
    -------
    Optional[str]
        Violation message if failed and raise_on_error=False, None if passed

    Raises
    ------
    DataValidationError
        If validation fails and raise_on_error=True
    """
    from src.core.exceptions import DataValidationError

    col = rule['column']
    valid_values = rule['valid_values']
    if col in df.columns:
        invalid_count = (~df[col].isin(valid_values)).sum()
        if invalid_count > 0:
            error_msg = f"Found {invalid_count} invalid buffer rates (expected: {valid_values})"
            if raise_on_error:
                raise DataValidationError(
                    error_msg,
                    validation_type="valid_buffer_rates",
                    business_impact="Invalid buffer values may cause incorrect product classification",
                    required_action="Verify buffer rate values against product specifications"
                )
            return error_msg
    return None


def _check_aggregation_ratio(
    rule: Dict[str, Any], raise_on_error: bool = True
) -> Optional[str]:
    """Check that aggregation ratio is within expected range.

    Parameters
    ----------
    rule : Dict[str, Any]
        Rule config with 'expected_min', 'expected_max', and 'actual' keys
    raise_on_error : bool, default=True
        If True, raise exception on failure. If False, return error string.

    Returns
    -------
    Optional[str]
        Violation message if failed and raise_on_error=False, None if passed

    Raises
    ------
    DataValidationError
        If validation fails and raise_on_error=True
    """
    from src.core.exceptions import DataValidationError

    expected_min = rule['expected_min']
    expected_max = rule['expected_max']
    actual = rule['actual']
    if not (expected_min <= actual <= expected_max):
        error_msg = f"Unusual aggregation ratio: {actual:.1f}x (expected {expected_min}-{expected_max}x)"
        if raise_on_error:
            raise DataValidationError(
                error_msg,
                validation_type="aggregation_ratio",
                business_impact="Data aggregation may have lost or duplicated records",
                required_action="Verify aggregation logic and source data cardinality"
            )
        return error_msg
    return None


def _check_sales_preservation(
    rule: Dict[str, Any], raise_on_error: bool = True
) -> Optional[str]:
    """Check that sales totals are preserved within tolerance.

    Parameters
    ----------
    rule : Dict[str, Any]
        Rule config with 'daily_total', 'weekly_total', and 'tolerance_pct' keys
    raise_on_error : bool, default=True
        If True, raise exception on failure. If False, return error string.

    Returns
    -------
    Optional[str]
        Violation message if failed and raise_on_error=False, None if passed

    Raises
    ------
    DataValidationError
        If validation fails and raise_on_error=True
    """
    from src.core.exceptions import DataValidationError

    daily_total = rule['daily_total']
    weekly_total = rule['weekly_total']
    tolerance_pct = rule['tolerance_pct']
    difference_pct = abs(daily_total - weekly_total) / daily_total * 100
    if difference_pct > tolerance_pct:
        error_msg = (
            f"Sales totals differ by {difference_pct:.2f}% (tolerance: {tolerance_pct:.1f}%). "
            f"Daily: ${daily_total:,.0f}, Weekly: ${weekly_total:,.0f}"
        )
        if raise_on_error:
            raise DataValidationError(
                error_msg,
                validation_type="sales_preservation",
                business_impact="Sales data inconsistency may produce incorrect elasticity estimates",
                required_action="Verify aggregation preserves sales totals"
            )
        return error_msg
    return None


def validate_business_rules(
    df: pd.DataFrame,
    rules: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = False
) -> List[str]:
    """Validate business-specific data quality rules.

    Args:
        df: DataFrame to validate
        rules: Dictionary of business rules to check
        raise_on_error: If True, raise on first violation. If False (default),
            collect all violations and return them.

    Returns:
        List of business rule violations (empty if all rules pass)

    Raises:
        DataValidationError: If raise_on_error=True and any rule fails
    """
    if rules is None:
        return []

    violations: List[str] = []

    # Map rule names to their validation functions
    # Pass raise_on_error=False to collect violations instead of raising
    rule_checks = [
        ('positive_premiums', lambda r: _check_positive_premiums(df, r, raise_on_error=raise_on_error)),
        ('date_consistency', lambda r: _check_date_consistency(df, r, raise_on_error=raise_on_error)),
        ('valid_buffer_rates', lambda r: _check_valid_buffer_rates(df, r, raise_on_error=raise_on_error)),
        ('aggregation_ratio', lambda r: _check_aggregation_ratio(r, raise_on_error=raise_on_error)),
        ('sales_preservation', lambda r: _check_sales_preservation(r, raise_on_error=raise_on_error)),
    ]

    for rule_name, check_fn in rule_checks:
        if rule_name in rules:
            violation = check_fn(rules[rule_name])
            if violation:
                violations.append(violation)

    return violations


# =============================================================================
# Status Determination Helpers
# =============================================================================

def _determine_validation_status(
    previous_metrics: Optional[ValidationMetrics],
    issues: List[str],
    warnings: List[str]
) -> str:
    """Determine overall validation status based on results.

    Args:
        previous_metrics: Metrics from previous run (None for first run)
        issues: List of critical issues
        warnings: List of warnings

    Returns:
        Status string: 'BASELINE' | 'FAILED' | 'WARNINGS' | 'PASSED'
    """
    if previous_metrics is None:
        return 'BASELINE'
    elif issues:
        return 'FAILED'
    elif warnings:
        return 'WARNINGS'
    else:
        return 'PASSED'


def _print_validation_summary(
    checkpoint_name: str,
    status: str,
    issues: List[str],
    warnings: List[str],
    current_metrics: ValidationMetrics,
    previous_metrics: Optional[ValidationMetrics],
    growth_pct: Optional[float],
    date_column: Optional[str]
) -> None:
    """Print validation summary to console.

    Args:
        checkpoint_name: Name of checkpoint
        status: Validation status
        issues: List of issues
        warnings: List of warnings
        current_metrics: Current run metrics
        previous_metrics: Previous run metrics
        growth_pct: Growth percentage if calculated
        date_column: Date column name if provided
    """
    if status == 'BASELINE':
        print(f"First production run - establishing baseline metadata for {checkpoint_name}")
    elif status == 'FAILED':
        print(f"[FAIL] Validation FAILED for {checkpoint_name}")
        for issue in issues:
            print(f"  Issue: {issue}")
    elif status == 'WARNINGS':
        print(f"[WARN] Validation passed with warnings for {checkpoint_name}")
        for warning in warnings:
            print(f"  Warning: {warning}")
    else:
        print(f"[PASS] Validation passed for {checkpoint_name}")

    print(f"  Schema stable: {current_metrics['column_count']} columns validated")

    if growth_pct is not None and previous_metrics is not None:
        prev_count = previous_metrics['record_count']
        curr_count = current_metrics['record_count']
        print(f"  Volume growth: {prev_count:,} → {curr_count:,} ({growth_pct:+.1f}%)")

    if date_column and current_metrics['min_date']:
        print(f"  Date range: {current_metrics['min_date']} to {current_metrics['max_date']}")


# =============================================================================
# Validation Stage Helpers
# =============================================================================

def _validate_schema_and_rules(
    df: pd.DataFrame,
    current_metrics: ValidationMetrics,
    previous_metrics: Optional[ValidationMetrics],
    config: ValidationConfig,
    business_rules: Optional[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """Validate schema stability and business rules.

    Args:
        df: DataFrame to validate
        current_metrics: Metrics from current run
        previous_metrics: Metrics from previous run (None for first run)
        config: Validation configuration
        business_rules: Optional business rules to validate

    Returns:
        Tuple of (issues, warnings) from validation

    Raises:
        ValueError: If critical schema mismatch or business rule violations occur
    """
    issues: List[str] = []
    warnings: List[str] = []

    # Validate schema stability
    schema_valid, schema_issues = validate_schema_stability(
        current_metrics,
        previous_metrics,
        strict=config['strict_schema'],
        critical_columns=config['critical_columns']
    )
    if not schema_valid:
        issues.extend(schema_issues)

    # Validate business rules
    if business_rules:
        rule_violations = validate_business_rules(df, business_rules)
        if rule_violations:
            critical_violations = [v for v in rule_violations if '[CRITICAL]' in v]
            if critical_violations:
                raise ValueError('\n'.join(critical_violations))
            warnings.extend(rule_violations)

    return issues, warnings


def _validate_patterns(
    current_metrics: ValidationMetrics,
    previous_metrics: Optional[ValidationMetrics],
    config: ValidationConfig,
    date_column: Optional[str]
) -> Tuple[List[str], Optional[float]]:
    """Validate growth patterns and date progression.

    Args:
        current_metrics: Metrics from current run
        previous_metrics: Metrics from previous run (None for first run)
        config: Validation configuration
        date_column: Optional name of date column for date range validation

    Returns:
        Tuple of (warnings, growth_percentage)
    """
    warnings: List[str] = []

    # Validate growth patterns
    _, growth_warnings, growth_pct = validate_growth_patterns(
        current_metrics,
        previous_metrics,
        config['growth_config']
    )
    warnings.extend(growth_warnings)

    # Validate date progression (if date column provided)
    if date_column:
        _, date_warnings = validate_date_range_progression(current_metrics, previous_metrics)
        warnings.extend(date_warnings)

    return warnings, growth_pct


def _finalize_validation(
    config: ValidationConfig,
    current_metrics: ValidationMetrics,
    previous_metrics: Optional[ValidationMetrics],
    issues: List[str],
    warnings: List[str],
    growth_pct: Optional[float],
    date_column: Optional[str]
) -> ValidationResult:
    """Save metrics and create validation result.

    Args:
        config: Validation configuration
        current_metrics: Metrics from current run
        previous_metrics: Metrics from previous run (None for first run)
        issues: List of critical issues
        warnings: List of warnings
        growth_pct: Growth percentage if calculated
        date_column: Date column name if provided

    Returns:
        ValidationResult with complete validation status
    """
    # Save current metrics
    metadata_path = save_validation_metadata(
        current_metrics,
        config['checkpoint_name'],
        config['version'],
        config['project_root']
    )

    # Determine status and print summary
    status = _determine_validation_status(previous_metrics, issues, warnings)
    _print_validation_summary(
        checkpoint_name=config['checkpoint_name'],
        status=status,
        issues=issues,
        warnings=warnings,
        current_metrics=current_metrics,
        previous_metrics=previous_metrics,
        growth_pct=growth_pct,
        date_column=date_column
    )

    return ValidationResult(
        checkpoint_name=config['checkpoint_name'],
        status=status,
        issues=issues,
        warnings=warnings,
        current_metrics=current_metrics,
        previous_metrics=previous_metrics,
        metadata_path=metadata_path
    )


# =============================================================================
# Main Validation Checkpoint
# =============================================================================

def run_production_validation_checkpoint(
    df: pd.DataFrame,
    config: ValidationConfig,
    date_column: Optional[str] = None,
    business_rules: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Run complete production validation checkpoint.

    Orchestrates all validation steps:
    1. Extract current metrics
    2. Load previous metrics
    3. Validate schema stability and business rules
    4. Validate growth and date patterns
    5. Save metrics and return results

    Args:
        df: DataFrame to validate
        config: Validation configuration
        date_column: Optional name of date column for date range validation
        business_rules: Optional business rules to validate

    Returns:
        ValidationResult with complete validation status

    Raises:
        ValueError: If critical validation failures occur (schema mismatch, critical business rules)
    """
    # Step 1: Extract current metrics
    current_metrics = extract_validation_metrics(df, date_column)

    # Step 2: Load previous metrics
    previous_metrics = load_previous_validation_metadata(
        config['checkpoint_name'],
        config['version'],
        config['project_root']
    )

    # Step 3: Validate schema stability and business rules
    issues, rule_warnings = _validate_schema_and_rules(
        df, current_metrics, previous_metrics, config, business_rules
    )

    # Step 4: Validate growth and date patterns
    pattern_warnings, growth_pct = _validate_patterns(
        current_metrics, previous_metrics, config, date_column
    )
    warnings = rule_warnings + pattern_warnings

    # Step 5: Save metrics and return results
    return _finalize_validation(
        config, current_metrics, previous_metrics,
        issues, warnings, growth_pct, date_column
    )
