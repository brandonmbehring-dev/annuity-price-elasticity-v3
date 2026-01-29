"""
Input Validation Functions for RILA Price Elasticity Pipeline.

This module provides canonical input validation functions that can be used
across multiple modules, eliminating duplicate validation logic.

CANONICAL VALIDATORS:
- validate_target_variable(): Unified target variable validation
- validate_required_string(): Generic string parameter validation
- validate_dataframe_column(): Generic column existence validation

Usage:
    from src.validation.input_validators import validate_target_variable

    # Raise exception if invalid (default)
    validate_target_variable(df, "sales_target_current")

    # Return warnings instead of raising
    warnings = validate_target_variable(df, "target", mode="warn")
"""

import pandas as pd
from typing import List, Optional, Union, Literal

# Validation modes
ValidationMode = Literal["raise", "warn"]


def validate_required_string(
    value: str,
    param_name: str,
    context: str = "",
    raise_on_error: bool = True,
) -> Optional[str]:
    """
    Validate that a value is a non-empty string.

    Parameters
    ----------
    value : str
        The value to validate
    param_name : str
        Name of the parameter (for error messages)
    context : str, optional
        Additional context for error messages
    raise_on_error : bool, default=True
        If True (default), raise exception on failure (fail-fast).
        If False, return error message string for warning collection.

    Returns
    -------
    Optional[str]
        Error message if validation fails and raise_on_error=False, None if valid

    Raises
    ------
    ValueError
        If value is not a non-empty string and raise_on_error=True
    """
    from src.core.exceptions import DataValidationError

    if not value or not isinstance(value, str):
        context_msg = f" {context}" if context else ""
        error_msg = (
            f"{param_name} must be non-empty string, got {value!r}.{context_msg}"
        )
        if raise_on_error:
            raise DataValidationError(
                error_msg,
                validation_type="required_string",
                business_impact=f"Cannot proceed without {param_name} definition",
                required_action=f"Specify valid {param_name} name"
            )
        return (
            f"CRITICAL: {error_msg} "
            f"Business impact: Cannot proceed without {param_name} definition. "
            f"Required action: Specify valid {param_name} name."
        )
    return None


def validate_dataframe_column(
    df: pd.DataFrame,
    column_name: str,
    context: str = "",
    require_numeric: bool = False,
    suggest_alternatives: bool = True,
    raise_on_error: bool = True,
) -> Optional[str]:
    """
    Validate that a column exists in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    column_name : str
        Column name to validate
    context : str, optional
        Additional context for error messages
    require_numeric : bool, default=False
        If True, also validate column is numeric dtype
    suggest_alternatives : bool, default=True
        If True, suggest similar column names in error
    raise_on_error : bool, default=True
        If True (default), raise exception on failure (fail-fast).
        If False, return error message string for warning collection.

    Returns
    -------
    Optional[str]
        Error message if validation fails and raise_on_error=False, None if valid

    Raises
    ------
    DataValidationError
        If validation fails and raise_on_error=True
    """
    from src.core.exceptions import DataValidationError

    if column_name not in df.columns:
        # Build helpful error message with alternatives
        context_msg = f" for {context}" if context else ""

        if suggest_alternatives:
            # Find similar columns (case-insensitive partial match)
            keyword = column_name.lower().split('_')[0] if '_' in column_name else column_name.lower()
            similar_cols = [col for col in df.columns if keyword in col.lower()][:5]
            alternatives = f"Similar columns: {similar_cols}. " if similar_cols else ""
        else:
            alternatives = ""

        error_msg = f"Column '{column_name}' not found in dataset{context_msg}. {alternatives}Available columns: {len(df.columns)} total."
        if raise_on_error:
            raise DataValidationError(
                error_msg,
                validation_type="column_exists",
                business_impact="Required column missing, analysis cannot proceed",
                required_action="Check column name configuration"
            )
        return f"CRITICAL: {error_msg} Check column name configuration."

    if require_numeric and not pd.api.types.is_numeric_dtype(df[column_name]):
        error_msg = f"Column '{column_name}' is not numeric (dtype: {df[column_name].dtype})."
        if raise_on_error:
            raise DataValidationError(
                error_msg,
                validation_type="column_numeric",
                business_impact="Numeric operations will fail",
                required_action="Ensure column contains numeric data"
            )
        return (
            f"CRITICAL: {error_msg} "
            f"Business impact: Numeric operations will fail. "
            f"Required action: Ensure column contains numeric data."
        )

    return None


def validate_target_variable(
    target_variable: str,
    df: Optional[pd.DataFrame] = None,
    mode: ValidationMode = "raise",
    require_numeric: bool = False,
    exception_type: type = ValueError,
) -> List[str]:
    """
    Validate target variable configuration and optionally its presence in data.

    Parameters
    ----------
    target_variable : str
        Target variable name to validate
    df : pd.DataFrame, optional
        If provided, validate column exists
    mode : {"raise", "warn"}
        Error handling mode. "raise" = fail-fast (default), "warn" = collect errors
    require_numeric : bool
        Validate column is numeric
    exception_type : type
        Exception type for "raise" mode (only used if raise_on_error would be True
        but we want a specific exception type)

    Returns
    -------
    List[str]
        List of warning/error messages (empty if valid)
    """
    from src.core.exceptions import DataValidationError

    # Determine whether to raise immediately or collect errors
    raise_on_error = (mode == "raise")
    errors: List[str] = []

    # 1. Validate target_variable is a non-empty string
    try:
        string_error = validate_required_string(
            target_variable,
            "target_variable",
            context="for regression analysis",
            raise_on_error=raise_on_error
        )
        if string_error:
            errors.append(string_error)
    except DataValidationError as e:
        # Re-raise with user's preferred exception type
        raise exception_type(str(e)) from e

    # 2. If DataFrame provided, validate column exists
    if df is not None and not errors:  # Only check if string validation passed
        try:
            column_error = validate_dataframe_column(
                df,
                target_variable,
                context="target variable validation",
                require_numeric=require_numeric,
                suggest_alternatives=True,
                raise_on_error=raise_on_error
            )
            if column_error:
                errors.append(column_error)
        except DataValidationError as e:
            # Re-raise with user's preferred exception type
            raise exception_type(str(e)) from e

    return errors


def validate_target_variable_string(target_variable: str) -> None:
    """
    Validate that target_variable is a non-empty string.

    Convenience wrapper for configuration validation (no DataFrame needed).
    Equivalent to previous _validate_target_variable in configuration_management.py.

    Parameters
    ----------
    target_variable : str
        Target variable name for regression analysis

    Raises
    ------
    ValueError
        If target_variable is empty or not a string
    """
    validate_target_variable(target_variable, df=None, mode="raise")


def validate_target_in_dataframe(
    df: pd.DataFrame,
    target_variable: str,
    require_numeric: bool = True,
) -> None:
    """
    Validate target variable exists in DataFrame.

    Convenience wrapper for data preprocessing validation.
    Equivalent to previous _validate_target_variable in data_preprocessing.py.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to check
    target_variable : str
        Name of target variable to validate

    Raises
    ------
    ValueError
        If target variable is not found
    """
    validate_target_variable(
        target_variable,
        df=df,
        mode="raise",
        require_numeric=require_numeric,
        exception_type=ValueError,
    )


def validate_target_with_warnings(
    df: pd.DataFrame,
    target_variable: str,
    require_numeric: bool = True,
) -> List[str]:
    """
    Validate target variable and return warnings (no exception).

    Convenience wrapper for pipeline orchestrator validation.
    Equivalent to previous _validate_target_variable in pipeline_orchestrator.py.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to check
    target_variable : str
        Target variable name to validate

    Returns
    -------
    List[str]
        List of warning messages (empty if valid)
    """
    return validate_target_variable(
        target_variable,
        df=df,
        mode="warn",
        require_numeric=require_numeric,
    )
