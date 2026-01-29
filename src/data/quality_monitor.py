"""
Data Quality Monitor for Data Preprocessing Pipeline

This module provides data quality monitoring capabilities specifically designed
for data preprocessing pipelines, focusing on operational data quality metrics
rather than scientific experimentation metrics.

Key Features:
- Data quality scoring at pipeline checkpoints
- Processing efficiency tracking
- Transformation lineage documentation
- DVC-compatible metrics output
- Business rule validation

Usage:
    from src.data.quality_monitor import DataQualityMonitor

    monitor = DataQualityMonitor("sales_processing_stage")
    quality_report = monitor.assess_data_quality(df)
    monitor.log_checkpoint_metrics(df, processing_time, memory_usage)
"""

import pandas as pd
import numpy as np
import time
import psutil
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class DataQualityReport:
    """Data quality assessment results for a pipeline checkpoint."""

    stage_name: str
    timestamp: str

    # Data Shape Metrics
    row_count: int
    column_count: int

    # Data Quality Metrics
    null_percentage: float
    duplicate_row_percentage: float
    data_type_consistency_score: float

    # Statistical Metrics
    numeric_columns_count: int
    categorical_columns_count: int
    outlier_percentage: float

    # Business Rule Validation
    business_rules_passed: int
    business_rules_total: int
    business_rules_score: float

    # Processing Metrics
    processing_time_seconds: float
    memory_usage_mb: float
    throughput_rows_per_second: float

    # Overall Quality Score (0-100)
    overall_quality_score: float

    # Validation Flags
    quality_threshold_met: bool
    ready_for_next_stage: bool


class DataQualityMonitor:
    """
    Data quality monitoring system for data preprocessing pipelines.

    Focuses on operational data quality metrics, processing efficiency,
    and business rule validation appropriate for data preparation stages.
    """

    def __init__(self, stage_name: str, quality_threshold: float = 85.0):
        """
        Initialize data quality monitor for a specific pipeline stage.

        Args:
            stage_name: Name of the pipeline stage being monitored
            quality_threshold: Minimum quality score required (0-100)
        """
        self.stage_name = stage_name
        self.quality_threshold = quality_threshold
        self.start_time = None
        self.start_memory = None

    def start_monitoring(self) -> None:
        """Start monitoring session - call before data processing."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage_mb()

    def _gather_processing_metrics(self) -> Dict[str, float]:
        """Gather processing time and memory metrics.

        Returns
        -------
        Dict[str, float]
            Processing metrics including time, memory, and throughput rate
        """
        end_time = time.time()
        processing_time = end_time - self.start_time
        current_memory = self._get_memory_usage_mb()
        memory_usage = current_memory - self.start_memory
        return {
            'processing_time': processing_time,
            'memory_usage': memory_usage
        }

    def _gather_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gather data quality and statistical metrics.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to assess

        Returns
        -------
        Dict[str, Any]
            Quality metrics including null%, duplicates, type consistency, outliers
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_percentage': self._calculate_null_percentage(df),
            'duplicate_percentage': self._calculate_duplicate_percentage(df),
            'type_consistency_score': self._assess_data_type_consistency(df),
            'numeric_columns_count': len(numeric_cols),
            'categorical_columns_count': len(categorical_cols),
            'outlier_percentage': self._calculate_outlier_percentage(df)
        }

    def _create_quality_report(
        self,
        quality_metrics: Dict[str, Any],
        processing_metrics: Dict[str, float],
        business_validation: Dict[str, Any],
        quality_score: float
    ) -> DataQualityReport:
        """Create DataQualityReport from collected metrics.

        Parameters
        ----------
        quality_metrics : Dict[str, Any]
            Data quality and statistical metrics
        processing_metrics : Dict[str, float]
            Processing time and memory metrics
        business_validation : Dict[str, Any]
            Business rule validation results
        quality_score : float
            Overall quality score (0-100)

        Returns
        -------
        DataQualityReport
            Complete quality report dataclass
        """
        throughput = quality_metrics['row_count'] / processing_metrics['processing_time'] \
            if processing_metrics['processing_time'] > 0 else 0

        return DataQualityReport(
            stage_name=self.stage_name,
            timestamp=datetime.now().isoformat(),
            row_count=quality_metrics['row_count'],
            column_count=quality_metrics['column_count'],
            null_percentage=quality_metrics['null_percentage'],
            duplicate_row_percentage=quality_metrics['duplicate_percentage'],
            data_type_consistency_score=quality_metrics['type_consistency_score'],
            numeric_columns_count=quality_metrics['numeric_columns_count'],
            categorical_columns_count=quality_metrics['categorical_columns_count'],
            outlier_percentage=quality_metrics['outlier_percentage'],
            business_rules_passed=business_validation['passed'],
            business_rules_total=business_validation['total'],
            business_rules_score=business_validation['score'],
            processing_time_seconds=processing_metrics['processing_time'],
            memory_usage_mb=processing_metrics['memory_usage'],
            throughput_rows_per_second=throughput,
            overall_quality_score=quality_score,
            quality_threshold_met=quality_score >= self.quality_threshold,
            ready_for_next_stage=quality_score >= self.quality_threshold
        )

    def assess_data_quality(self, df: pd.DataFrame,
                          business_rules: Optional[List[Dict[str, Any]]] = None) -> DataQualityReport:
        """
        Comprehensive data quality assessment for preprocessing pipeline.

        Args:
            df: DataFrame to assess
            business_rules: List of business rules to validate

        Returns:
            DataQualityReport with comprehensive quality metrics
        """
        if self.start_time is None:
            raise ValueError("Must call start_monitoring() before assessment")

        # Step 1: Gather processing metrics
        processing_metrics = self._gather_processing_metrics()

        # Step 2: Gather quality metrics
        quality_metrics = self._gather_quality_metrics(df)

        # Step 3: Validate business rules
        business_validation = self._validate_business_rules(df, business_rules or [])

        # Step 4: Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(
            quality_metrics['null_percentage'],
            quality_metrics['duplicate_percentage'],
            quality_metrics['type_consistency_score'],
            business_validation['score'],
            quality_metrics['outlier_percentage']
        )

        # Step 5: Create and return quality report
        return self._create_quality_report(
            quality_metrics, processing_metrics, business_validation, quality_score
        )

    def log_checkpoint_metrics(self, df: pd.DataFrame,
                             business_rules: Optional[List[Dict[str, Any]]] = None,
                             output_path: str = "metrics.json") -> DataQualityReport:
        """
        Log data quality metrics to DVC-compatible metrics file.

        Args:
            df: DataFrame to assess and log
            business_rules: Business rules to validate
            output_path: Path to metrics.json file

        Returns:
            DataQualityReport generated
        """
        report = self.assess_data_quality(df, business_rules)

        # Load existing metrics or create new structure
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # Add data quality metrics for this stage
        stage_key = f"data_quality_{self.stage_name}"
        metrics[stage_key] = {
            "overall_quality_score": report.overall_quality_score,
            "null_percentage": report.null_percentage,
            "processing_time_seconds": report.processing_time_seconds,
            "throughput_rows_per_second": report.throughput_rows_per_second,
            "memory_usage_mb": report.memory_usage_mb,
            "row_count": report.row_count,
            "column_count": report.column_count,
            "quality_threshold_met": report.quality_threshold_met,
            "business_rules_score": report.business_rules_score,
            "timestamp": report.timestamp
        }

        # Save updated metrics
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"[OK] Data quality metrics logged for {self.stage_name}")
        print(f"   Quality Score: {report.overall_quality_score:.1f}/100")
        print(f"   Processing Time: {report.processing_time_seconds:.2f}s")
        print(f"   Throughput: {report.throughput_rows_per_second:.0f} rows/sec")
        print(f"   Ready for Next Stage: {'YES' if report.ready_for_next_stage else 'NO'}")

        return report

    def validate_quality_gate(self, df: pd.DataFrame,
                            business_rules: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Validate data quality gate - fail pipeline if quality insufficient.

        Args:
            df: DataFrame to validate
            business_rules: Business rules to validate

        Returns:
            True if quality gate passed, False otherwise

        Raises:
            DataQualityError: If quality gate fails with detailed context
        """
        report = self.assess_data_quality(df, business_rules)

        if not report.quality_threshold_met:
            raise DataQualityError(
                f"QUALITY GATE FAILED for {self.stage_name}. "
                f"Quality score {report.overall_quality_score:.1f} below threshold {self.quality_threshold}. "
                f"Issues: Null data {report.null_percentage:.1f}%, "
                f"Business rules {report.business_rules_score:.1f}%, "
                f"Duplicates {report.duplicate_row_percentage:.1f}%. "
                f"Pipeline cannot continue with poor data quality."
            )

        return True

    def _calculate_null_percentage(self, df: pd.DataFrame) -> float:
        """Calculate percentage of null values across entire DataFrame."""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        return (null_cells / total_cells) * 100 if total_cells > 0 else 0

    def _calculate_duplicate_percentage(self, df: pd.DataFrame) -> float:
        """Calculate percentage of duplicate rows."""
        duplicates = df.duplicated().sum()
        return (duplicates / len(df)) * 100 if len(df) > 0 else 0

    def _assess_data_type_consistency(self, df: pd.DataFrame) -> float:
        """Assess data type consistency across columns."""
        if len(df.columns) == 0:
            return 100.0

        consistent_columns = 0

        for col in df.columns:
            # Check if column has consistent data type
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                consistent_columns += 1  # Empty columns are "consistent"
                continue

            # For numeric columns, check if all values are actually numeric
            if df[col].dtype in ['int64', 'float64']:
                try:
                    pd.to_numeric(non_null_values, errors='raise')
                    consistent_columns += 1
                except (ValueError, TypeError):
                    pass  # Inconsistent numeric column
            else:
                consistent_columns += 1  # Non-numeric columns assumed consistent

        return (consistent_columns / len(df.columns)) * 100

    def _calculate_outlier_percentage(self, df: pd.DataFrame) -> float:
        """Calculate percentage of outlier values using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0

        total_numeric_values = 0
        total_outliers = 0

        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) < 4:  # Need at least 4 values for IQR
                continue

            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((values < lower_bound) | (values > upper_bound)).sum()

            total_numeric_values += len(values)
            total_outliers += outliers

        return (total_outliers / total_numeric_values) * 100 if total_numeric_values > 0 else 0

    def _validate_business_rules(self, df: pd.DataFrame,
                               business_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate business rules against DataFrame. Returns dict with passed/total/score/failures."""
        if not business_rules:
            return {'passed': 0, 'total': 0, 'score': 100.0, 'failures': []}

        passed = 0
        failures = []

        for rule in business_rules:
            try:
                rule_name = rule['name']
                condition = rule['condition']
                error_msg = rule.get('error_msg', f"Business rule '{rule_name}' failed")

                # Evaluate condition (should return boolean or boolean Series)
                result = eval(condition, {'df': df, 'pd': pd, 'np': np})

                if isinstance(result, pd.Series):
                    # If Series, check if all values are True
                    rule_passed = result.all()
                else:
                    # If scalar boolean
                    rule_passed = bool(result)

                if rule_passed:
                    passed += 1
                else:
                    failures.append({'rule': rule_name, 'message': error_msg})

            except Exception as e:
                failures.append({'rule': rule.get('name', 'unknown'),
                               'message': f"Rule evaluation error: {e}"})

        total = len(business_rules)
        score = (passed / total) * 100 if total > 0 else 100.0

        return {
            'passed': passed,
            'total': total,
            'score': score,
            'failures': failures
        }

    def _calculate_overall_quality_score(self, null_pct: float, duplicate_pct: float,
                                       type_consistency: float, business_score: float,
                                       outlier_pct: float) -> float:
        """
        Calculate overall data quality score (0-100).

        Weighted combination of various quality metrics appropriate for data preprocessing.
        """
        # Convert percentages to scores (lower is better for null, duplicate, outlier)
        null_score = max(0, 100 - null_pct * 2)  # Penalize null values heavily
        duplicate_score = max(0, 100 - duplicate_pct * 1.5)  # Moderate penalty for duplicates
        outlier_score = max(0, 100 - outlier_pct * 0.5)  # Light penalty for outliers

        # Weighted average (business rules most important for data preprocessing)
        weights = {
            'business_rules': 0.35,  # Business logic compliance most critical
            'null_data': 0.25,      # Data completeness very important
            'type_consistency': 0.20, # Data type correctness important
            'duplicates': 0.15,      # Duplicate detection moderately important
            'outliers': 0.05         # Outlier detection least critical for preprocessing
        }

        overall_score = (
            weights['business_rules'] * business_score +
            weights['null_data'] * null_score +
            weights['type_consistency'] * type_consistency +
            weights['duplicates'] * duplicate_score +
            weights['outliers'] * outlier_score
        )

        return min(100.0, max(0.0, overall_score))

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class DataQualityError(Exception):
    """Exception raised when data quality requirements are not met."""
    pass


# Predefined business rule templates for common data preprocessing validations
class BusinessRuleTemplates:
    """Common business rule templates for data preprocessing pipelines."""

    @staticmethod
    def non_empty_dataset() -> Dict[str, Any]:
        """Rule: Dataset must not be empty."""
        return {
            'name': 'non_empty_dataset',
            'condition': 'len(df) > 0',
            'error_msg': 'Dataset is empty - no rows to process'
        }

    @staticmethod
    def required_columns(columns: List[str]) -> Dict[str, Any]:
        """Rule: Required columns must be present."""
        cols_str = str(columns)
        return {
            'name': 'required_columns_present',
            'condition': f'all(col in df.columns for col in {cols_str})',
            'error_msg': f'Required columns missing: {columns}'
        }

    @staticmethod
    def no_all_null_columns() -> Dict[str, Any]:
        """Rule: No columns should be entirely null."""
        return {
            'name': 'no_all_null_columns',
            'condition': '~(df.isnull().all()).any()',
            'error_msg': 'Some columns are entirely null'
        }

    @staticmethod
    def positive_values_only(columns: List[str]) -> Dict[str, Any]:
        """Rule: Specified columns must contain only positive values."""
        condition_parts = [f'(df["{col}"].dropna() > 0).all()' for col in columns]
        condition = ' and '.join(condition_parts)
        return {
            'name': 'positive_values_only',
            'condition': condition,
            'error_msg': f'Negative values found in columns: {columns}'
        }

    @staticmethod
    def date_range_validation(date_column: str, min_date: str, max_date: str) -> Dict[str, Any]:
        """Rule: Date column must be within specified range."""
        return {
            'name': 'date_range_validation',
            'condition': f'df["{date_column}"].dt.date.between(pd.to_datetime("{min_date}").date(), pd.to_datetime("{max_date}").date()).all()',
            'error_msg': f'Dates in {date_column} outside valid range [{min_date}, {max_date}]'
        }


# Convenience function for quick quality assessment
def assess_data_quality_quick(df: pd.DataFrame, stage_name: str) -> DataQualityReport:
    """
    Quick data quality assessment with default settings.

    Args:
        df: DataFrame to assess
        stage_name: Name of pipeline stage

    Returns:
        DataQualityReport with quality metrics
    """
    monitor = DataQualityMonitor(stage_name)
    monitor.start_monitoring()
    return monitor.assess_data_quality(df)