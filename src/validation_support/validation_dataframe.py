"""
DataFrame Transformation Validation Module for Mathematical Equivalence.

This module contains the DataFrameEquivalenceValidator class and related
functions for validating data pipeline transformations preserve mathematical
precision and business logic.

Module Responsibilities:
- DataFrame transformation equivalence validation
- Refactoring equivalence enforcement
- Pipeline stage validation
- Baseline comparison functions
- Business impact assessment

Used by: mathematical_equivalence.py (re-exports for backward compatibility)
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.validation_support.validation_constants import (
    TOLERANCE,
    EquivalenceValidationResult,
    EquivalenceResult,
    MathematicalEquivalenceError,
)

# Configure logging
logger = logging.getLogger(__name__)

# MLflow availability
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# =============================================================================
# DATAFRAME TRANSFORMATION VALIDATION
# =============================================================================


class DataFrameEquivalenceValidator:
    """
    Mathematical equivalence validator for data pipeline transformations.

    Ensures that data pipeline refactoring preserves mathematical precision
    and business logic, preventing loss of validated work.

    Usage:
        validator = DataFrameEquivalenceValidator()
        result = validator.validate_transformation_equivalence(
            original_df, transformed_df, "weekly_aggregation"
        )
    """

    def __init__(self, default_tolerance: float = TOLERANCE):
        """
        Initialize validator with default tolerance.

        Args:
            default_tolerance: Default precision tolerance (target: 1e-12)
        """
        self.default_tolerance = default_tolerance
        self.validation_history: List[EquivalenceValidationResult] = []

    def _build_validation_result(
        self, transformation_name: str, timestamp: str, tolerance: float,
        original_df: pd.DataFrame, transformed_df: pd.DataFrame,
        shapes_match: bool, comparison: Dict[str, Any], business_impact: Dict[str, str]
    ) -> EquivalenceValidationResult:
        """Build EquivalenceValidationResult from comparison data."""
        mathematically_equivalent = (
            shapes_match and comparison['values_equivalent'] and
            comparison['max_absolute_difference'] <= tolerance
        )
        return EquivalenceValidationResult(
            transformation_name=transformation_name, timestamp=timestamp, tolerance=tolerance,
            shapes_match=shapes_match, original_shape=original_df.shape,
            transformed_shape=transformed_df.shape, values_equivalent=comparison['values_equivalent'],
            max_absolute_difference=comparison['max_absolute_difference'],
            max_relative_difference=comparison['max_relative_difference'],
            columns_compared=comparison['columns_compared'],
            equivalent_columns=comparison['equivalent_columns'],
            non_equivalent_columns=comparison['non_equivalent_columns'],
            mathematically_equivalent=mathematically_equivalent, validation_passed=mathematically_equivalent,
            business_impact_assessment=business_impact['assessment'], recommendation=business_impact['recommendation']
        )

    def validate_transformation_equivalence(
        self, original_df: pd.DataFrame, transformed_df: pd.DataFrame,
        transformation_name: str, tolerance: Optional[float] = None,
        ignore_columns: Optional[List[str]] = None, log_to_mlflow: bool = True
    ) -> EquivalenceValidationResult:
        """Validate mathematical equivalence between original and transformed DataFrames."""
        tolerance = tolerance if tolerance is not None else self.default_tolerance
        ignore_columns = ignore_columns or []
        timestamp = datetime.now().isoformat()

        shapes_match = original_df.shape == transformed_df.shape
        columns_match = set(original_df.columns) == set(transformed_df.columns)

        comparison = (self._compare_dataframe_contents(original_df, transformed_df, tolerance, ignore_columns)
                     if shapes_match and columns_match
                     else self._build_shape_mismatch_result(original_df.shape, transformed_df.shape))

        mathematically_equivalent = (shapes_match and comparison['values_equivalent'] and
                                    comparison['max_absolute_difference'] <= tolerance)
        business_impact = self._assess_business_impact(mathematically_equivalent, comparison, transformation_name)

        result = self._build_validation_result(
            transformation_name, timestamp, tolerance, original_df, transformed_df,
            shapes_match, comparison, business_impact
        )

        if log_to_mlflow and MLFLOW_AVAILABLE:
            self._log_to_mlflow(result)
        self.validation_history.append(result)
        return result

    def validate_refactoring_equivalence(
        self,
        original_results: Union[pd.DataFrame, Dict[str, Any]],
        refactored_results: Union[pd.DataFrame, Dict[str, Any]],
        operation_name: str,
        tolerance: Optional[float] = None
    ) -> EquivalenceValidationResult:
        """
        Validate that refactoring preserves mathematical equivalence.

        Args:
            original_results: Results from original implementation
            refactored_results: Results from refactored implementation
            operation_name: Name of the refactored operation
            tolerance: Numerical tolerance (default: 1e-12)

        Returns:
            EquivalenceValidationResult with refactoring validation
        """
        if isinstance(original_results, dict):
            original_df = pd.DataFrame([original_results])
        else:
            original_df = original_results.copy()

        if isinstance(refactored_results, dict):
            refactored_df = pd.DataFrame([refactored_results])
        else:
            refactored_df = refactored_results.copy()

        return self.validate_transformation_equivalence(
            original_df=original_df,
            transformed_df=refactored_df,
            transformation_name=f"REFACTORING_{operation_name}",
            tolerance=tolerance
        )

    def enforce_equivalence_requirement(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        transformation_name: str,
        tolerance: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Enforce mathematical equivalence with fail-fast behavior.

        Args:
            original_df: Original DataFrame
            transformed_df: Transformed DataFrame
            transformation_name: Name of transformation
            tolerance: Numerical tolerance

        Returns:
            Transformed DataFrame if validation passes

        Raises:
            MathematicalEquivalenceError: If validation fails
        """
        result = self.validate_transformation_equivalence(
            original_df, transformed_df, transformation_name, tolerance
        )

        if not result.validation_passed:
            raise MathematicalEquivalenceError(
                f"Mathematical equivalence validation FAILED for {transformation_name}. "
                f"{result.business_impact_assessment} "
                f"Max difference: {result.max_absolute_difference:.2e} exceeds tolerance {result.tolerance:.2e}."
            )

        return transformed_df

    def _build_shape_mismatch_result(
        self, original_shape: Tuple[int, int], transformed_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Build comparison result for shape/column mismatch case."""
        return {
            'values_equivalent': False,
            'max_absolute_difference': float('inf'),
            'max_relative_difference': float('inf'),
            'columns_compared': 0,
            'equivalent_columns': [],
            'non_equivalent_columns': [{
                'column': 'SHAPE_MISMATCH',
                'max_abs_diff': float('inf'),
                'issue': f"Shape mismatch: {original_shape} vs {transformed_shape}"
            }]
        }

    def _compare_dataframe_contents(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        tolerance: float,
        ignore_columns: List[str]
    ) -> Dict[str, Any]:
        """Compare DataFrame contents with specified tolerance."""
        equivalent_columns: List[str] = []
        non_equivalent_columns: List[Dict[str, Any]] = []
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        columns_compared = 0

        for col in df1.columns:
            if col in ignore_columns:
                continue

            columns_compared += 1
            is_eq, err_info, col_abs, col_rel = self._compare_single_column(
                df1, df2, col, tolerance
            )

            if is_eq:
                equivalent_columns.append(col)
            else:
                non_equivalent_columns.append(err_info)

            max_abs_diff = max(max_abs_diff, col_abs)
            max_rel_diff = max(max_rel_diff, col_rel)

        return {
            'values_equivalent': len(non_equivalent_columns) == 0,
            'max_absolute_difference': max_abs_diff,
            'max_relative_difference': max_rel_diff,
            'columns_compared': columns_compared,
            'equivalent_columns': equivalent_columns,
            'non_equivalent_columns': non_equivalent_columns
        }

    def _compare_single_column(
        self, df1: pd.DataFrame, df2: pd.DataFrame, col: str, tolerance: float
    ) -> Tuple[bool, Optional[Dict[str, Any]], float, float]:
        """Compare a single column, handling type detection."""
        try:
            # Categorical comparison
            if df1[col].dtype in ['object', 'category'] and df2[col].dtype in ['object', 'category']:
                if df1[col].equals(df2[col]):
                    return True, None, 0.0, 0.0
                return False, {
                    'column': col,
                    'issue': 'Categorical values differ'
                }, float('inf'), 0.0

            # Numerical comparison
            if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                return self._compare_numerical_column(df1, df2, col, tolerance)

            # Type mismatch
            return False, {
                'column': col,
                'issue': f'Type mismatch: {df1[col].dtype} vs {df2[col].dtype}'
            }, float('inf'), 0.0

        except Exception as e:
            return False, {
                'column': col,
                'issue': f'Comparison error: {str(e)}'
            }, float('inf'), 0.0

    def _compare_numerical_column(
        self, df1: pd.DataFrame, df2: pd.DataFrame, col: str, tolerance: float
    ) -> Tuple[bool, Optional[Dict[str, Any]], float, float]:
        """Compare numerical column with tolerance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            mask1, mask2 = pd.isna(df1[col]), pd.isna(df2[col])

            if not mask1.equals(mask2):
                return False, {
                    'column': col,
                    'issue': 'NaN patterns differ'
                }, float('inf'), float('inf')

            valid_mask = ~mask1 & ~mask2
            if valid_mask.sum() == 0:
                return True, None, 0.0, 0.0

            val1, val2 = df1[col][valid_mask], df2[col][valid_mask]
            abs_diff = np.abs(val1 - val2)
            col_max_abs = abs_diff.max()
            rel_diff = np.where(np.abs(val1) > 1e-15, abs_diff / np.abs(val1), abs_diff)
            col_max_rel = rel_diff.max()

            if col_max_abs <= tolerance:
                return True, None, col_max_abs, col_max_rel

            return False, {
                'column': col,
                'max_abs_diff': float(col_max_abs),
                'issue': f'Exceeds tolerance: {col_max_abs:.2e} > {tolerance:.2e}'
            }, col_max_abs, col_max_rel

    def _assess_business_impact(
        self, mathematically_equivalent: bool, comparison: Dict[str, Any], name: str
    ) -> Dict[str, str]:
        """Assess business impact of mathematical equivalence validation."""
        if mathematically_equivalent:
            return {
                'assessment': f"VALIDATION PASSED: {name} preserves mathematical precision. "
                              f"All {comparison['columns_compared']} columns equivalent within tolerance.",
                'recommendation': "Transformation is safe to deploy."
            }

        non_equiv_count = len(comparison['non_equivalent_columns'])
        max_diff = comparison['max_absolute_difference']

        if max_diff == float('inf') or max_diff > 1e-6:
            impact_level = "CRITICAL"
            recommendation = "DO NOT DEPLOY: Critical mathematical differences detected."
        elif max_diff > 1e-9:
            impact_level = "MODERATE"
            recommendation = "REVIEW REQUIRED: Moderate differences may affect business logic."
        else:
            impact_level = "MINOR"
            recommendation = "DOCUMENT: Minor differences detected."

        return {
            'assessment': f"VALIDATION FAILED: {name} breaks mathematical equivalence. "
                          f"{non_equiv_count} columns differ. Max: {max_diff:.2e}. Impact: {impact_level}",
            'recommendation': recommendation
        }

    def _log_to_mlflow(self, result: EquivalenceValidationResult) -> None:
        """Log validation results to MLflow for tracking."""
        try:
            with mlflow.start_run(nested=True, run_name=f"equivalence_{result.transformation_name}"):
                mlflow.log_param("transformation_name", result.transformation_name)
                mlflow.log_param("tolerance", result.tolerance)
                mlflow.log_metric("mathematically_equivalent", 1.0 if result.mathematically_equivalent else 0.0)
                mlflow.log_metric("max_absolute_difference",
                                  result.max_absolute_difference if result.max_absolute_difference != float('inf') else -1)
                mlflow.log_metric("columns_compared", result.columns_compared)
                mlflow.set_tag("validation_type", "mathematical_equivalence")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def generate_validation_report(self, output_path: str = "equivalence_report.md") -> str:
        """Generate comprehensive validation report from validation history."""
        if not self.validation_history:
            raise ValueError("No validation history available. Run validations first.")

        passed = sum(1 for v in self.validation_history if v.validation_passed)
        total = len(self.validation_history)

        report_lines = [
            "# Mathematical Equivalence Validation Report",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Validations Performed**: {total}",
            f"**Passed**: {passed}/{total} ({passed/total*100:.1f}%)",
            "",
            "## Detailed Results",
            ""
        ]

        for i, v in enumerate(self.validation_history, 1):
            status = "PASS" if v.validation_passed else "FAIL"
            report_lines.extend([
                f"### {i}. [{status}] {v.transformation_name}",
                "",
                f"- **Tolerance**: {v.tolerance:.2e}",
                f"- **Max Difference**: {v.max_absolute_difference:.2e}",
                f"- **Shapes Match**: {'Yes' if v.shapes_match else 'No'}",
                f"- **Columns Compared**: {v.columns_compared}",
                f"- **Assessment**: {v.business_impact_assessment}",
                ""
            ])

        with open(output_path, 'w') as f:
            f.write("\n".join(report_lines))

        return output_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_pipeline_stage_equivalence(
    original_df: pd.DataFrame,
    refactored_df: pd.DataFrame,
    stage_name: str,
    tolerance: float = TOLERANCE
) -> EquivalenceValidationResult:
    """
    Quick pipeline stage equivalence validation.

    Args:
        original_df: Original stage output
        refactored_df: Refactored stage output
        stage_name: Pipeline stage name
        tolerance: Numerical tolerance

    Returns:
        EquivalenceValidationResult
    """
    validator = DataFrameEquivalenceValidator(tolerance)
    return validator.validate_refactoring_equivalence(
        original_df, refactored_df, stage_name, tolerance
    )


def enforce_transformation_equivalence(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    transformation_name: str,
    tolerance: float = TOLERANCE
) -> pd.DataFrame:
    """
    Enforce transformation equivalence with fail-fast validation.

    Args:
        original_df: Original DataFrame
        transformed_df: Transformed DataFrame
        transformation_name: Transformation name
        tolerance: Numerical tolerance

    Returns:
        Transformed DataFrame if valid

    Raises:
        MathematicalEquivalenceError: If validation fails
    """
    validator = DataFrameEquivalenceValidator(tolerance)
    return validator.enforce_equivalence_requirement(
        original_df, transformed_df, transformation_name, tolerance
    )


def _interpret_equivalence(
    validation_passed: bool, max_abs_diff: float,
    tolerance: float, target_tolerance: float
) -> str:
    """Generate business interpretation of equivalence validation."""
    if validation_passed and max_abs_diff <= target_tolerance:
        return f"EXCELLENT: Perfect equivalence (diff: {max_abs_diff:.2e})"
    elif validation_passed:
        return f"GOOD: Equivalence within tolerance (diff: {max_abs_diff:.2e} <= {tolerance:.2e})"
    return f"CRITICAL: Differences exceed tolerance (diff: {max_abs_diff:.2e} > {tolerance:.2e})"


def _aggregate_comparison_results(
    baseline: Dict[str, Any], original: Dict[str, Any], tolerance: float
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Aggregate comparison results from AIC and model comparisons."""
    differences: List[Dict[str, Any]] = []
    max_abs, max_rel = 0.0, 0.0

    if 'aic_results' in baseline and 'aic_results' in original:
        cmp = _compare_dataframes_for_equivalence(
            baseline['aic_results'], original['aic_results'], 'aic_results', tolerance
        )
        differences.extend(cmp['differences'])
        max_abs = max(max_abs, cmp['max_absolute_difference'])
        max_rel = max(max_rel, cmp['max_relative_difference'])

    if 'selected_model' in baseline and 'selected_model' in original:
        cmp = _compare_models_for_equivalence(
            baseline['selected_model'], original['selected_model'], tolerance
        )
        differences.extend(cmp['differences'])
        max_abs = max(max_abs, cmp['max_absolute_difference'])
        max_rel = max(max_rel, cmp['max_relative_difference'])

    return differences, max_abs, max_rel


def validate_baseline_equivalence(
    baseline_results: Dict[str, Any],
    original_results: Dict[str, Any],
    tolerance: float = TOLERANCE,
    target_tolerance: float = 0.0
) -> EquivalenceResult:
    """Validate mathematical equivalence between baseline and original methodologies."""
    logger.info(f"Validating baseline equivalence with tolerance {tolerance}")

    differences, max_abs_diff, max_rel_diff = _aggregate_comparison_results(
        baseline_results, original_results, tolerance
    )

    validation_passed = max_abs_diff <= tolerance
    return EquivalenceResult(
        comparison_type="baseline_vs_original",
        validation_passed=validation_passed,
        tolerance_used=tolerance,
        differences_found=differences,
        max_absolute_difference=max_abs_diff,
        max_relative_difference=max_rel_diff,
        summary_metrics={
            'total_comparisons': len(differences),
            'max_absolute_difference': max_abs_diff,
            'tolerance_ratio': max_abs_diff / tolerance if tolerance > 0 else np.inf
        },
        business_interpretation=_interpret_equivalence(
            validation_passed, max_abs_diff, tolerance, target_tolerance
        ),
        remediation_required=not validation_passed or max_abs_diff > target_tolerance,
        remediation_suggestions=_generate_suggestions(differences, max_abs_diff, tolerance)
    )


def _compare_dataframes_for_equivalence(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name: str,
    tolerance: float
) -> Dict[str, Any]:
    """Compare two DataFrames for equivalence."""
    result: Dict[str, Any] = {'differences': [], 'max_absolute_difference': 0.0, 'max_relative_difference': 0.0}

    if df1.shape != df2.shape:
        result['differences'].append({
            'comparison_type': f'{name}_shape',
            'baseline_shape': df1.shape,
            'original_shape': df2.shape,
            'exceeds_tolerance': True
        })
        result['max_absolute_difference'] = float('inf')
        return result

    for col in df1.columns:
        if col not in df2.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df1[col]):
            continue

        abs_diffs = np.abs(df1[col].values - df2[col].values)
        max_diff = np.nanmax(abs_diffs)
        result['max_absolute_difference'] = max(result['max_absolute_difference'], max_diff)

        if max_diff > tolerance:
            result['differences'].append({
                'comparison_type': f'{name}_{col}',
                'max_difference': max_diff,
                'exceeds_tolerance': True
            })

    return result


def _compare_models_for_equivalence(
    model1: Dict[str, Any],
    model2: Dict[str, Any],
    tolerance: float
) -> Dict[str, Any]:
    """Compare two model dictionaries for equivalence."""
    result: Dict[str, Any] = {'differences': [], 'max_absolute_difference': 0.0, 'max_relative_difference': 0.0}

    # Compare features
    if model1.get('features') != model2.get('features'):
        result['differences'].append({
            'comparison_type': 'selected_model_features',
            'baseline_features': model1.get('features'),
            'original_features': model2.get('features'),
            'exceeds_tolerance': True
        })

    # Compare numerical metrics
    for metric in ['aic', 'r_squared', 'n_features']:
        if metric in model1 and metric in model2:
            diff = abs(model1[metric] - model2[metric])
            result['max_absolute_difference'] = max(result['max_absolute_difference'], diff)
            if diff > tolerance:
                result['differences'].append({
                    'comparison_type': f'model_{metric}',
                    'difference': diff,
                    'exceeds_tolerance': True
                })

    return result


def _generate_suggestions(
    differences: List[Dict[str, Any]],
    max_diff: float,
    tolerance: float
) -> List[str]:
    """Generate remediation suggestions based on differences."""
    if max_diff <= tolerance:
        return ["No remediation required - validation passed"]

    suggestions: List[str] = []
    critical = [d for d in differences if 'features' in d.get('comparison_type', '')]
    if critical:
        suggestions.append("CRITICAL: Fix feature selection logic - different features selected")

    if max_diff > 1e-6:
        suggestions.append("Review floating-point arithmetic and numerical precision")

    return suggestions if suggestions else ["Review implementation for numerical precision issues"]
