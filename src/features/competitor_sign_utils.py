#!/usr/bin/env python3
"""
Functional Competitor Sign Utilities

Pure functional utilities for transforming competitor column signs in DataFrames.
All functions are immutable operations that return new DataFrames following
unified coding standards and functional programming principles.

This module provides utilities for flipping signs of competitor columns in
financial datasets, particularly for economic theory adjustments where higher
competitor rates should positively correlate with product sales.

Examples
--------
>>> import pandas as pd
>>> from competitor_sign_utils import flip_competitor_signs
>>>
>>> df = pd.DataFrame({
...     'sales': [100, 200, 300],
...     'competitor_mid_t2': [1.5, 2.0, 2.5],
...     'competitor_top5_t3': [3.0, 3.5, 4.0],
...     'other_feature': [10, 20, 30]
... })
>>>
>>> flipped_df = flip_competitor_signs(df, 'competitor_')
>>> print(flipped_df['competitor_mid_t2'])  # Now negative values

Author: Generated following UNIFIED_CODING_STANDARDS.md
Date: 2025-10-22
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


def flip_competitor_signs(df: pd.DataFrame, column_pattern: str) -> pd.DataFrame:
    """Flip signs of columns matching pattern; returns new DataFrame."""
    # Input validation with comprehensive error handling
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    if not isinstance(column_pattern, str):
        raise TypeError(f"Expected str for column_pattern, got {type(column_pattern).__name__}")

    if df.empty:
        raise ValueError("Cannot process empty DataFrame for competitor sign flipping")

    if not column_pattern.strip():
        raise ValueError("column_pattern cannot be empty string")

    try:
        # Create immutable copy of original DataFrame
        result_df = df.copy()

        # Identify competitor columns using functional filtering
        competitor_columns = [col for col in result_df.columns
                            if column_pattern.lower() in col.lower()]

        if not competitor_columns:
            # Return copy even if no changes needed (functional purity)
            return result_df

        # Apply sign flip transformation using vectorized operations
        for col in competitor_columns:
            result_df[col] = -result_df[col]

        return result_df

    except Exception as e:
        raise ValueError(
            f"Failed to flip competitor signs with pattern '{column_pattern}' "
            f"on DataFrame shape {df.shape}: {e}"
        ) from e


def validate_competitor_columns(df: pd.DataFrame, column_pattern: str) -> Tuple[List[str], bool]:
    """Validate and identify competitor columns matching pattern."""
    # Input validation following unified standards
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    if not isinstance(column_pattern, str):
        raise TypeError(f"Expected str for column_pattern, got {type(column_pattern).__name__}")

    if df.empty:
        raise ValueError("Cannot validate empty DataFrame")

    if not column_pattern.strip():
        raise ValueError("column_pattern cannot be empty string")

    try:
        # Functional approach to identify matching columns
        matching_columns = [col for col in df.columns
                          if column_pattern.lower() in col.lower()]

        # Validation logic for competitor columns
        is_valid = True
        validation_issues = []

        for col in matching_columns:
            # Check if column contains numeric data
            if not pd.api.types.is_numeric_dtype(df[col]):
                is_valid = False
                validation_issues.append(f"Column {col} is not numeric")

            # Check for all NaN values
            if df[col].isna().all():
                is_valid = False
                validation_issues.append(f"Column {col} contains all NaN values")

        # Return validation results as immutable tuple
        return matching_columns, is_valid

    except Exception as e:
        raise ValueError(
            f"Failed to validate competitor columns with pattern '{column_pattern}': {e}"
        ) from e


def flip_multiple_patterns(df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
    """Apply sign flipping to multiple column patterns using functional composition."""
    # Input validation with comprehensive error handling
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    if not isinstance(patterns, list):
        raise TypeError(f"Expected list for patterns, got {type(patterns).__name__}")

    if df.empty:
        raise ValueError("Cannot process empty DataFrame")

    if not patterns:
        raise ValueError("patterns list cannot be empty")

    # Validate all patterns are strings
    if not all(isinstance(pattern, str) for pattern in patterns):
        invalid_types = [type(p).__name__ for p in patterns if not isinstance(p, str)]
        raise TypeError(f"All patterns must be strings, found: {invalid_types}")

    try:
        # Functional composition: start with original DataFrame copy
        result_df = df.copy()

        # Apply each pattern transformation sequentially
        for pattern in patterns:
            if pattern.strip():  # Skip empty patterns
                result_df = flip_competitor_signs(result_df, pattern)

        return result_df

    except Exception as e:
        raise ValueError(
            f"Failed to apply multiple pattern flips {patterns} "
            f"on DataFrame shape {df.shape}: {e}"
        ) from e


def _validate_flip_summary_inputs(
    df_original: pd.DataFrame,
    df_flipped: pd.DataFrame,
    column_pattern: str
) -> None:
    """Validate inputs for flip summary creation."""
    if not isinstance(df_original, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame for df_original, got {type(df_original).__name__}")
    if not isinstance(df_flipped, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame for df_flipped, got {type(df_flipped).__name__}")
    if not isinstance(column_pattern, str):
        raise TypeError(f"Expected str for column_pattern, got {type(column_pattern).__name__}")
    if df_original.empty or df_flipped.empty:
        raise ValueError("Cannot create summary from empty DataFrames")
    if df_original.shape != df_flipped.shape:
        raise ValueError(f"DataFrame shapes must match: {df_original.shape} vs {df_flipped.shape}")


def create_flip_summary(
    df_original: pd.DataFrame,
    df_flipped: pd.DataFrame,
    column_pattern: str
) -> pd.DataFrame:
    """
    Create summary DataFrame comparing original and flipped competitor columns.

    Parameters
    ----------
    df_original : pd.DataFrame
        Original DataFrame before sign flipping
    df_flipped : pd.DataFrame
        Transformed DataFrame after sign flipping
    column_pattern : str
        Pattern used to identify competitor columns

    Returns
    -------
    pd.DataFrame
        Summary with 'column_name', 'original_mean', 'flipped_mean',
        'sign_changed', 'row_count'
    """
    _validate_flip_summary_inputs(df_original, df_flipped, column_pattern)

    try:
        competitor_columns = [col for col in df_original.columns
                              if column_pattern.lower() in col.lower()]

        if not competitor_columns:
            return pd.DataFrame(columns=['column_name', 'original_mean', 'flipped_mean',
                                         'sign_changed', 'row_count'])

        summary_data = []
        for col in competitor_columns:
            summary_data.append({
                'column_name': col,
                'original_mean': df_original[col].mean(),
                'flipped_mean': df_flipped[col].mean(),
                'sign_changed': np.sign(df_original[col].mean()) != np.sign(df_flipped[col].mean()),
                'row_count': len(df_original)
            })

        return pd.DataFrame(summary_data)

    except Exception as e:
        raise ValueError(f"Failed to create flip summary for pattern '{column_pattern}': {e}") from e


# Demonstration and testing functions
def demonstrate_functional_usage() -> None:
    """
    Demonstrate functional usage of competitor sign utilities.

    This function showcases the pure functional approach and proper usage
    patterns for all utility functions in this module.
    """
    print("=" * 60)
    print("FUNCTIONAL COMPETITOR SIGN UTILITIES DEMONSTRATION")
    print("=" * 60)

    # Create sample dataset mimicking RILA structure
    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5, freq='W'),
        'sales_target_current': [80000000, 85000000, 90000000, 88000000, 92000000],
        'competitor_mid_t2': [2.5, 2.3, 2.7, 2.4, 2.6],
        'competitor_top5_t3': [3.2, 3.0, 3.4, 3.1, 3.3],
        'rival_median_t1': [2.8, 2.6, 2.9, 2.7, 2.8],
        'prudential_rate_current': [2.0, 2.1, 2.0, 2.1, 2.0]
    })

    print(f"Original DataFrame shape: {sample_data.shape}")
    print("Original competitor columns:")
    competitor_cols, is_valid = validate_competitor_columns(sample_data, 'competitor_')
    print(f"  Found columns: {competitor_cols}")
    print(f"  Validation passed: {is_valid}")

    # Demonstrate single pattern flipping
    print("\n1. Single Pattern Flipping:")
    flipped_data = flip_competitor_signs(sample_data, 'competitor_')
    print(f"  competitor_mid_t2 mean: {sample_data['competitor_mid_t2'].mean():.3f} -> {flipped_data['competitor_mid_t2'].mean():.3f}")

    # Demonstrate multiple pattern flipping
    print("\n2. Multiple Pattern Flipping:")
    multi_flipped = flip_multiple_patterns(sample_data, ['competitor_', 'rival_'])
    print(f"  Patterns processed: ['competitor_', 'rival_']")
    print(f"  rival_median_t1 mean: {sample_data['rival_median_t1'].mean():.3f} -> {multi_flipped['rival_median_t1'].mean():.3f}")

    # Demonstrate summary creation
    print("\n3. Summary Analysis:")
    summary = create_flip_summary(sample_data, flipped_data, 'competitor_')
    print(summary[['column_name', 'original_mean', 'flipped_mean', 'sign_changed']].to_string(index=False))

    print(f"\n4. Functional Purity Verification:")
    print(f"  Original DataFrame unchanged: {sample_data['competitor_mid_t2'].mean():.3f}")
    print(f"  All operations return new DataFrames - no mutations")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    demonstrate_functional_usage()