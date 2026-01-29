"""
Schema Validator - Dataset Input/Output Validation

This module provides dataset schema validation to replace mathematical precision validation.
Focuses on practical data quality checks and schema conformance for daily-changing data.

CANONICAL FUNCTIONS:
- validate_input_schema(): Validate incoming dataset structure
- validate_output_schema(): Validate pipeline output datasets
- create_schema_report(): Generate validation reports

REPLACEMENT STRATEGY:
- Replaces 1e-12 mathematical precision validation
- Focuses on schema conformance for daily data changes
- Provides business-relevant data quality checks

Usage Pattern (from notebooks):
    from src.data.schema_validator import validate_input_schema, validate_output_schema

    # Replace mathematical precision validation
    validate_input_schema(df, "sales_data")
    processed_df = pipeline_function(df)
    validate_output_schema(processed_df, "processed_sales", expected_columns=['date', 'sales'])
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# CONTEXT ANCHOR: SCHEMA VALIDATION OBJECTIVES
# =============================================================================
# PURPOSE: Replace mathematical precision validation with practical schema validation for daily-changing data
# USED BY: All pipeline notebooks to validate input/output datasets at strategic boundaries
# DEPENDENCIES: pandas, logging (no external schema libraries for simplicity)
# LAST VALIDATED: 2025-11-21 (initial creation for mathematical validation replacement)
# PATTERN STATUS: CANONICAL (replaces all mathematical precision validation in notebooks)
#
# ARCHITECTURAL FLOW: dataset → schema_check → business_rules → validation_report
# SUCCESS CRITERIA: Practical validation that works with daily-changing data, no 1e-12 precision requirements
# INTEGRATION: Works with existing pipeline structure, replaces validation checkpoints
# MAINTENANCE: Update schemas as business requirements change, monitor validation failures

logger = logging.getLogger(__name__)

class DatasetSchema:
    """Define expected schema for dataset validation."""

    def __init__(self,
                 name: str,
                 required_columns: List[str],
                 optional_columns: List[str] = None,
                 column_types: Dict[str, str] = None,
                 business_rules: Dict[str, Dict] = None):
        """
        Initialize dataset schema definition.

        Args:
            name: Schema name for identification
            required_columns: Columns that must be present
            optional_columns: Columns that may be present
            column_types: Expected data types (pandas dtype strings)
            business_rules: Business validation rules (min/max values, ranges)
        """
        self.name = name
        self.required_columns = required_columns
        self.optional_columns = optional_columns or []
        self.column_types = column_types or {}
        self.business_rules = business_rules or {}

class SchemaValidator:
    """Dataset schema validation for pipeline boundaries."""

    def __init__(self):
        self.validation_history = []
        self.schemas = self._initialize_schemas()

    def _create_sales_data_schema(self) -> DatasetSchema:
        """Create schema for sales data."""
        return DatasetSchema(
            name='sales_data',
            required_columns=['application_signed_date', 'contract_issue_date', 'product_name', 'sales_amount'],
            optional_columns=['term_filter', 'buffer_rate_filter'],
            column_types={
                'application_signed_date': 'datetime64[ns]',
                'contract_issue_date': 'datetime64[ns]',
                'sales_amount': 'float64'
            },
            business_rules={
                'sales_amount': {'min_value': 0, 'max_value': 100_000_000},
                'row_count': {'min_rows': 1000}  # Business minimum for reliable analysis
            }
        )

    def _create_time_series_schema(self) -> DatasetSchema:
        """Create schema for time series data."""
        return DatasetSchema(
            name='time_series_data',
            required_columns=['date', 'sales'],
            column_types={
                'date': 'datetime64[ns]',
                'sales': 'float64'
            },
            business_rules={
                'sales': {'min_value': 0},
                'row_count': {'min_rows': 100, 'max_rows': 3000},
                'date_continuity': {'max_gap_days': 10}
            }
        )

    def _create_competitive_rates_schema(self) -> DatasetSchema:
        """Create schema for competitive rates data."""
        return DatasetSchema(
            name='competitive_rates',
            required_columns=['date', 'Prudential'],
            optional_columns=['Allianz', 'Athene', 'Brighthouse', 'Equitable', 'Jackson', 'Lincoln'],
            column_types={
                'date': 'datetime64[ns]',
                'Prudential': 'float64'
            },
            business_rules={
                'Prudential': {'min_value': 0.001, 'max_value': 0.20},  # 0.1% to 20% rates
                'row_count': {'min_rows': 1000}
            }
        )

    def _create_weekly_aggregated_schema(self) -> DatasetSchema:
        """Create schema for weekly aggregated data."""
        return DatasetSchema(
            name='weekly_aggregated',
            required_columns=['date', 'sales'],
            optional_columns=['C_weighted_mean', 'DGS5', 'VIX'],
            column_types={
                'date': 'datetime64[ns]',
                'sales': 'float64'
            },
            business_rules={
                'row_count': {'min_rows': 50, 'max_rows': 500},
                'sales': {'min_value': 0}
            }
        )

    def _create_final_dataset_schema(self) -> DatasetSchema:
        """Create schema for final dataset."""
        return DatasetSchema(
            name='final_dataset',
            required_columns=['date', 'sales', 'Spread', 'sales_log'],
            column_types={
                'date': 'datetime64[ns]',
                'sales': 'float64',
                'Spread': 'float64',
                'sales_log': 'float64'
            },
            business_rules={
                'row_count': {'min_rows': 150, 'max_rows': 180},
                'feature_count': {'min_features': 590},
                'sales': {'min_value': 0},
                'Spread': {'min_value': -500, 'max_value': 500}  # Basis points range
            }
        )

    def _initialize_schemas(self) -> Dict[str, DatasetSchema]:
        """Initialize known schemas for common datasets."""
        return {
            'sales_data': self._create_sales_data_schema(),
            'time_series_data': self._create_time_series_schema(),
            'competitive_rates': self._create_competitive_rates_schema(),
            'weekly_aggregated': self._create_weekly_aggregated_schema(),
            'final_dataset': self._create_final_dataset_schema()
        }

    def _validate_required_columns(
        self, df: pd.DataFrame, schema: DatasetSchema, result: Dict[str, Any]
    ) -> None:
        """Check required columns and emptiness.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        schema : DatasetSchema
            Schema with required columns
        result : Dict[str, Any]
            Validation result to update in-place
        """
        if df.empty:
            result['errors'].append("Dataset is empty")
            result['is_valid'] = False

        missing_required = [col for col in schema.required_columns if col not in df.columns]
        if missing_required:
            result['errors'].append(f"Missing required columns: {missing_required}")
            result['is_valid'] = False

    def _validate_column_types(
        self, df: pd.DataFrame, schema: DatasetSchema, result: Dict[str, Any]
    ) -> None:
        """Validate column types against schema expectations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        schema : DatasetSchema
            Schema with expected column types
        result : Dict[str, Any]
            Validation result to update in-place
        """
        for col, expected_type in schema.column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type.startswith('datetime') and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    result['warnings'].append(f"Column {col}: expected datetime, got {actual_type}")
                elif expected_type == 'float64' and not pd.api.types.is_numeric_dtype(df[col]):
                    result['warnings'].append(f"Column {col}: expected numeric, got {actual_type}")

    def _apply_business_rules_validation(
        self, df: pd.DataFrame, schema: DatasetSchema, result: Dict[str, Any]
    ) -> None:
        """Apply business rule validation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        schema : DatasetSchema
            Schema with business rules
        result : Dict[str, Any]
            Validation result to update in-place
        """
        for rule_name, rule_config in schema.business_rules.items():
            if rule_name == 'row_count':
                row_count = len(df)
                if 'min_rows' in rule_config and row_count < rule_config['min_rows']:
                    result['errors'].append(
                        f"Insufficient rows: {row_count} < {rule_config['min_rows']} (business minimum)"
                    )
                    result['is_valid'] = False
                if 'max_rows' in rule_config and row_count > rule_config['max_rows']:
                    result['warnings'].append(
                        f"High row count: {row_count} > {rule_config['max_rows']} (check for data duplication)"
                    )
            elif rule_name in df.columns:
                self._validate_column_value_range(df, rule_name, rule_config, result)

    def _validate_column_value_range(
        self, df: pd.DataFrame, col_name: str, rule_config: Dict, result: Dict[str, Any]
    ) -> None:
        """Validate column value ranges against business rules.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        col_name : str
            Column name to check
        rule_config : Dict
            Rule configuration with min_value/max_value
        result : Dict[str, Any]
            Validation result to update in-place
        """
        col_data = df[col_name]
        if 'min_value' in rule_config:
            min_val = col_data.min()
            if pd.notna(min_val) and min_val < rule_config['min_value']:
                result['warnings'].append(
                    f"Column {col_name}: minimum value {min_val} < {rule_config['min_value']}"
                )
        if 'max_value' in rule_config:
            max_val = col_data.max()
            if pd.notna(max_val) and max_val > rule_config['max_value']:
                result['warnings'].append(
                    f"Column {col_name}: maximum value {max_val} > {rule_config['max_value']}"
                )

    def _generate_validation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate validation summary statistics.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that was validated

        Returns
        -------
        Dict[str, Any]
            Summary statistics including row/column counts, missing values, duplicates
        """
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }

    def _log_validation_results(
        self, schema_name: str, df: pd.DataFrame, result: Dict[str, Any]
    ) -> None:
        """Log validation results to logger.

        Parameters
        ----------
        schema_name : str
            Name of schema being validated
        df : pd.DataFrame
            DataFrame that was validated
        result : Dict[str, Any]
            Validation result with errors and warnings
        """
        status = "PASSED" if result['is_valid'] else "FAILED"
        logger.info(f"Schema validation {status}: {schema_name} ({len(df)} rows, {len(df.columns)} columns)")

        for warning in result['warnings']:
            logger.warning(f"Schema validation warning: {warning}")

        for error in result['errors']:
            logger.error(f"Schema validation error: {error}")

    def validate_input_schema(self,
                            df: pd.DataFrame,
                            schema_name: str,
                            context: str = "") -> Dict[str, Any]:
        """
        Validate input dataset against expected schema.

        Args:
            df: DataFrame to validate
            schema_name: Name of schema to validate against
            context: Additional context for error reporting

        Returns:
            Validation result dictionary

        Raises:
            ValueError: If critical validation failures occur
        """
        if schema_name not in self.schemas:
            logger.warning(f"Unknown schema: {schema_name}. Performing basic validation only.")
            return self._basic_validation(df, schema_name, context)

        schema = self.schemas[schema_name]
        validation_result = {
            'schema_name': schema_name,
            'context': context,
            'timestamp': datetime.now(),
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'summary': {}
        }

        try:
            # Step 1: Validate required columns and emptiness
            self._validate_required_columns(df, schema, validation_result)

            # Step 2: Validate column types
            self._validate_column_types(df, schema, validation_result)

            # Step 3: Apply business rules
            self._apply_business_rules_validation(df, schema, validation_result)

            # Step 4: Generate summary
            validation_result['summary'] = self._generate_validation_summary(df)

            # Step 5: Log results
            self._log_validation_results(schema_name, df, validation_result)

            self.validation_history.append(validation_result)
            return validation_result

        except Exception as e:
            logger.error(f"Schema validation error for {schema_name}: {str(e)}")
            validation_result['errors'].append(f"Validation process failed: {str(e)}")
            validation_result['is_valid'] = False
            raise

    def validate_output_schema(self,
                             df: pd.DataFrame,
                             schema_name: str,
                             expected_columns: List[str] = None,
                             context: str = "") -> Dict[str, Any]:
        """
        Validate output dataset with additional expected columns.

        Args:
            df: DataFrame to validate
            schema_name: Base schema name
            expected_columns: Additional columns expected in output
            context: Pipeline context for reporting

        Returns:
            Validation result dictionary
        """
        # Use base schema validation
        result = self.validate_input_schema(df, schema_name, context)

        # Additional output-specific checks
        if expected_columns:
            missing_expected = [col for col in expected_columns if col not in df.columns]
            if missing_expected:
                result['warnings'].append(f"Expected output columns not found: {missing_expected}")

        return result

    def _basic_validation(self, df: pd.DataFrame, schema_name: str, context: str) -> Dict[str, Any]:
        """Basic validation for unknown schemas."""
        return {
            'schema_name': schema_name,
            'context': context,
            'timestamp': datetime.now(),
            'is_valid': not df.empty,
            'warnings': [] if not df.empty else ["Dataset is empty"],
            'errors': [] if not df.empty else ["Empty dataset - no schema validation possible"],
            'summary': {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().sum() if not df.empty else 0,
                'duplicate_rows': df.duplicated().sum() if not df.empty else 0
            }
        }

    def _format_report_summary(self, recent: List[Dict[str, Any]]) -> List[str]:
        """Format summary statistics for report.

        Parameters
        ----------
        recent : List[Dict[str, Any]]
            List of recent validation results

        Returns
        -------
        List[str]
            Formatted summary lines
        """
        passed_count = sum(1 for v in recent if v['is_valid'])
        warning_count = sum(len(v['warnings']) for v in recent)
        error_count = sum(len(v['errors']) for v in recent)

        return [
            "SUMMARY:",
            f"  [PASS] Passed: {passed_count}/{len(recent)}",
            f"  [WARN] Warnings: {warning_count}",
            f"  [FAIL] Errors: {error_count}",
            ""
        ]

    def _format_individual_validation(self, validation: Dict[str, Any]) -> List[str]:
        """Format a single validation result for report.

        Parameters
        ----------
        validation : Dict[str, Any]
            Single validation result

        Returns
        -------
        List[str]
            Formatted lines for this validation
        """
        status_icon = "[PASS]" if validation['is_valid'] else "[FAIL]"
        lines = [
            f"{status_icon} {validation['schema_name']} ({validation['context']})",
            f"    Rows: {validation['summary']['rows']:,}, Columns: {validation['summary']['columns']}",
            f"    Missing values: {validation['summary']['missing_values']:,}",
        ]

        for warning in validation['warnings'][:3]:
            lines.append(f"    [WARN] {warning}")

        for error in validation['errors'][:3]:
            lines.append(f"    [ERROR] {error}")

        lines.append("")
        return lines

    def create_schema_report(self, recent_validations: int = 10) -> str:
        """
        Generate schema validation report for recent validations.

        Args:
            recent_validations: Number of recent validations to include

        Returns:
            Formatted validation report
        """
        if not self.validation_history:
            return "No schema validations performed yet."

        recent = self.validation_history[-recent_validations:]

        report_lines = [
            "SCHEMA VALIDATION REPORT",
            "=" * 50,
            f"Recent validations: {len(recent)}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        report_lines.extend(self._format_report_summary(recent))

        for validation in recent:
            report_lines.extend(self._format_individual_validation(validation))

        return "\n".join(report_lines)

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
# - Notebooks: Use convenience functions (validate_input_schema, etc.)
# - Tests: Use create_validator() for test isolation
# - Internal src/: Pass validators explicitly when needed

# Global validator instance for notebook convenience
_schema_validator = None

def get_schema_validator() -> SchemaValidator:
    """Get schema validator instance.

    MIGRATION NOTE: This function now returns the singleton for backward compatibility.
    New code should create validators explicitly:
        validator = SchemaValidator()

    Returns:
        SchemaValidator instance (singleton for backward compatibility)
    """
    global _schema_validator
    if _schema_validator is None:
        _schema_validator = SchemaValidator()
    return _schema_validator

def create_validator() -> SchemaValidator:
    """Create a new schema validator instance (DI pattern).

    This is the preferred pattern for dependency injection.
    Use this when you need an isolated validator instance (e.g., in tests).

    Returns:
        New SchemaValidator instance

    Example:
        validator = create_validator()
        result = validate_input_schema(df, "sales_data", validator=validator)
    """
    return SchemaValidator()

# Convenience functions for notebook use (backward-compatible with DI support)
def validate_input_schema(
    df: pd.DataFrame,
    schema_name: str,
    context: str = "",
    validator: Optional[SchemaValidator] = None
) -> Dict[str, Any]:
    """Validate input dataset schema (convenience function).

    Args:
        df: DataFrame to validate
        schema_name: Name of schema to validate against
        context: Optional context string for error messages
        validator: Optional validator instance (DI pattern). If None, uses singleton.

    Returns:
        Validation result dictionary
    """
    if validator is None:
        validator = get_schema_validator()  # Fallback to singleton
    return validator.validate_input_schema(df, schema_name, context)

def validate_output_schema(
    df: pd.DataFrame,
    schema_name: str,
    expected_columns: List[str] = None,
    context: str = "",
    validator: Optional[SchemaValidator] = None
) -> Dict[str, Any]:
    """Validate output dataset schema (convenience function).

    Args:
        df: DataFrame to validate
        schema_name: Name of schema to validate against
        expected_columns: Optional list of expected columns
        context: Optional context string for error messages
        validator: Optional validator instance (DI pattern). If None, uses singleton.

    Returns:
        Validation result dictionary
    """
    if validator is None:
        validator = get_schema_validator()  # Fallback to singleton
    return validator.validate_output_schema(df, schema_name, expected_columns, context)

def create_schema_report(
    recent_validations: int = 10,
    validator: Optional[SchemaValidator] = None
) -> str:
    """Generate schema validation report (convenience function).

    Args:
        recent_validations: Number of recent validations to include
        validator: Optional validator instance (DI pattern). If None, uses singleton.

    Returns:
        Formatted validation report string
    """
    if validator is None:
        validator = get_schema_validator()  # Fallback to singleton
    return validator.create_schema_report(recent_validations)

# Validation shortcuts for common patterns (DI-compatible)
def validate_pipeline_input(
    df: pd.DataFrame,
    pipeline_stage: str = None,
    pipeline_name: str = None,
    validator: Optional[SchemaValidator] = None
) -> Dict[str, Any]:
    """Quick validation for pipeline inputs with standard error handling.

    Args:
        df: DataFrame to validate
        pipeline_stage: Pipeline stage name (preferred)
        pipeline_name: Pipeline name (legacy parameter)
        validator: Optional validator instance (DI pattern). If None, uses singleton.

    Returns:
        Validation result dictionary
    """
    # Support both pipeline_stage (new) and pipeline_name (old) parameter names
    name = pipeline_stage or pipeline_name
    if not name:
        raise ValueError("Must provide either pipeline_stage or pipeline_name parameter")

    # Determine schema based on pipeline name/stage
    if 'sales' in name.lower():
        schema_name = 'sales_data'
    elif 'time_series' in name.lower():
        schema_name = 'time_series_data'
    elif 'competitive' in name.lower() or 'wink' in name.lower():
        schema_name = 'competitive_rates'
    else:
        schema_name = 'basic'  # Will use basic validation

    return validate_input_schema(df, schema_name, f"Pipeline: {name}", validator=validator)

def validate_pipeline_output(
    df: pd.DataFrame,
    pipeline_stage: str = None,
    pipeline_name: str = None,
    expected_columns: List[str] = None,
    validator: Optional[SchemaValidator] = None
) -> Dict[str, Any]:
    """Quick validation for pipeline outputs with standard error handling.

    Args:
        df: DataFrame to validate
        pipeline_stage: Pipeline stage name (preferred)
        pipeline_name: Pipeline name (legacy parameter)
        expected_columns: Optional list of expected columns
        validator: Optional validator instance (DI pattern). If None, uses singleton.

    Returns:
        Validation result dictionary
    """
    # Support both pipeline_stage (new) and pipeline_name (old) parameter names
    name = pipeline_stage or pipeline_name
    if not name:
        raise ValueError("Must provide either pipeline_stage or pipeline_name parameter")

    # Determine schema based on pipeline name/stage
    if 'final' in name.lower():
        schema_name = 'final_dataset'
    elif 'weekly' in name.lower():
        schema_name = 'weekly_aggregated'
    elif 'time_series' in name.lower():
        schema_name = 'time_series_data'
    else:
        schema_name = 'basic'

    return validate_output_schema(df, schema_name, expected_columns, f"Pipeline output: {name}", validator=validator)