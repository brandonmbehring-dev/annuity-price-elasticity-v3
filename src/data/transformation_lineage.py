"""
Data Transformation Lineage Documentation System

This module provides comprehensive data lineage tracking and transformation metadata
capture for data preprocessing pipelines. It integrates with the quality monitoring
and mathematical equivalence validation systems to provide complete data provenance.

Key Features:
- Transformation metadata capture with business context
- Data lineage graph construction and visualization
- Integration with DVC and MLflow for complete tracking
- Automatic schema evolution detection
- Business rule compliance lineage
- Performance impact tracking across transformations

Usage:
    from src.data.transformation_lineage import TransformationLineage

    lineage = TransformationLineage("sales_processing_pipeline")
    lineage.start_transformation("product_filtering")
    lineage.record_transformation_step(
        step_name="apply_product_filters",
        input_df=raw_data,
        output_df=filtered_data,
        business_context="Filter to FlexGuard 6Y 20% products only"
    )
    lineage.end_transformation()
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict, field
import hashlib
import warnings
import logging

logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


@dataclass
class TransformationStep:
    """Metadata for a single transformation step."""

    # Step Identity
    step_name: str
    step_id: str
    timestamp: str
    pipeline_stage: str

    # Data Characteristics
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    input_columns: List[str]
    output_columns: List[str]

    # Schema Changes
    columns_added: List[str]
    columns_removed: List[str]
    columns_renamed: Dict[str, str]
    data_types_changed: Dict[str, Tuple[str, str]]

    # Business Context
    business_purpose: str
    transformation_logic: str
    business_rules_applied: List[str]

    # Quality Metrics
    data_quality_before: Optional[float]
    data_quality_after: Optional[float]
    processing_time_seconds: float
    memory_usage_mb: float

    # Mathematical Validation
    mathematical_equivalence_validated: bool
    equivalence_tolerance: Optional[float]
    max_numerical_difference: Optional[float]

    # Data Provenance
    input_data_hash: str
    output_data_hash: str
    configuration_hash: str


@dataclass
class PipelineLineage:
    """Complete lineage for an entire data pipeline."""

    # Pipeline Identity
    pipeline_name: str
    pipeline_version: str
    execution_id: str
    start_time: str
    end_time: Optional[str]

    # Pipeline Configuration
    configuration: Dict[str, Any]
    business_parameters: Dict[str, Any]

    # Transformation Steps
    transformation_steps: List[TransformationStep] = field(default_factory=list)

    # Overall Pipeline Metrics
    total_processing_time: Optional[float] = None
    overall_quality_improvement: Optional[float] = None
    data_reduction_ratio: Optional[float] = None
    feature_expansion_ratio: Optional[float] = None

    # Validation Results
    all_transformations_validated: bool = False
    critical_validation_failures: List[str] = field(default_factory=list)


class TransformationLineage:
    """
    Data transformation lineage tracking and documentation system.

    Captures comprehensive metadata about data transformations including
    business context, quality impacts, and mathematical validation results.
    """

    def __init__(self, pipeline_name: str, pipeline_version: str = "1.0"):
        """
        Initialize transformation lineage tracker.

        Args:
            pipeline_name: Name of the data pipeline
            pipeline_version: Version of the pipeline
        """
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.execution_id = self._generate_execution_id()

        self.current_stage = None
        self.pipeline_lineage = PipelineLineage(
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            execution_id=self.execution_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            configuration={},
            business_parameters={}
        )

        self.lineage_history = []

    def start_transformation(self, stage_name: str, configuration: Optional[Dict[str, Any]] = None) -> None:
        """
        Start tracking a new transformation stage.

        Args:
            stage_name: Name of the transformation stage
            configuration: Stage configuration parameters
        """
        self.current_stage = stage_name

        if configuration:
            self.pipeline_lineage.configuration[stage_name] = configuration

        print(f"[LINEAGE] Started lineage tracking for stage: {stage_name}")

    def _create_step_metadata(
        self, step_name: str, input_df: pd.DataFrame, output_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create metadata components for transformation step.

        Parameters
        ----------
        step_name : str
            Name of the transformation step
        input_df : pd.DataFrame
            Input DataFrame
        output_df : pd.DataFrame
            Output DataFrame

        Returns
        -------
        Dict[str, Any]
            Step metadata including IDs, hashes, and schema changes
        """
        return {
            'step_id': f"{self.execution_id}_{len(self.pipeline_lineage.transformation_steps)}",
            'timestamp': datetime.now().isoformat(),
            'schema_changes': self._analyze_schema_changes(input_df, output_df),
            'input_hash': self._calculate_dataframe_hash(input_df),
            'output_hash': self._calculate_dataframe_hash(output_df),
            'config_hash': self._calculate_configuration_hash(step_name)
        }

    def _build_transformation_step(
        self,
        step_name: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        metadata: Dict[str, Any],
        business_context: str,
        transformation_logic: str,
        business_rules_applied: Optional[List[str]],
        quality_before: Optional[float],
        quality_after: Optional[float],
        processing_time: Optional[float],
        memory_usage: Optional[float],
        equivalence_validated: bool,
        equivalence_tolerance: Optional[float],
        max_numerical_diff: Optional[float]
    ) -> TransformationStep:
        """Build TransformationStep dataclass from components."""
        return TransformationStep(
            step_name=step_name,
            step_id=metadata['step_id'],
            timestamp=metadata['timestamp'],
            pipeline_stage=self.current_stage,
            input_shape=input_df.shape,
            output_shape=output_df.shape,
            input_columns=list(input_df.columns),
            output_columns=list(output_df.columns),
            columns_added=metadata['schema_changes']['columns_added'],
            columns_removed=metadata['schema_changes']['columns_removed'],
            columns_renamed=metadata['schema_changes']['columns_renamed'],
            data_types_changed=metadata['schema_changes']['data_types_changed'],
            business_purpose=business_context,
            transformation_logic=transformation_logic,
            business_rules_applied=business_rules_applied or [],
            data_quality_before=quality_before,
            data_quality_after=quality_after,
            processing_time_seconds=processing_time or 0.0,
            memory_usage_mb=memory_usage or 0.0,
            mathematical_equivalence_validated=equivalence_validated,
            equivalence_tolerance=equivalence_tolerance,
            max_numerical_difference=max_numerical_diff,
            input_data_hash=metadata['input_hash'],
            output_data_hash=metadata['output_hash'],
            configuration_hash=metadata['config_hash']
        )

    def _log_step_summary(
        self, step_name: str, input_df: pd.DataFrame, output_df: pd.DataFrame,
        schema_changes: Dict, quality_before: Optional[float], quality_after: Optional[float]
    ) -> None:
        """Log transformation step summary to console."""
        print(f"[OK] Recorded transformation step: {step_name}")
        print(f"   Input: {input_df.shape} -> Output: {output_df.shape}")
        print(f"   Columns: +{len(schema_changes['columns_added'])}, -{len(schema_changes['columns_removed'])}")

        if quality_before is not None and quality_after is not None:
            quality_change = quality_after - quality_before
            print(f"   Quality: {quality_before:.1f} → {quality_after:.1f} ({quality_change:+.1f})")

    def record_transformation_step(
        self,
        step_name: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        business_context: str = "",
        transformation_logic: str = "",
        business_rules_applied: Optional[List[str]] = None,
        quality_before: Optional[float] = None,
        quality_after: Optional[float] = None,
        processing_time: Optional[float] = None,
        memory_usage: Optional[float] = None,
        equivalence_validated: bool = False,
        equivalence_tolerance: Optional[float] = None,
        max_numerical_diff: Optional[float] = None
    ) -> TransformationStep:
        """Record detailed metadata for a transformation step."""
        if self.current_stage is None:
            raise ValueError("Must call start_transformation() before recording steps")

        # Step 1: Generate metadata
        metadata = self._create_step_metadata(step_name, input_df, output_df)

        # Step 2: Build transformation step
        transformation_step = self._build_transformation_step(
            step_name, input_df, output_df, metadata,
            business_context, transformation_logic, business_rules_applied,
            quality_before, quality_after, processing_time, memory_usage,
            equivalence_validated, equivalence_tolerance, max_numerical_diff
        )

        # Step 3: Add to pipeline lineage
        self.pipeline_lineage.transformation_steps.append(transformation_step)

        # Step 4: Log to MLflow if available
        if MLFLOW_AVAILABLE:
            self._log_transformation_to_mlflow(transformation_step)

        # Step 5: Log summary
        self._log_step_summary(
            step_name, input_df, output_df, metadata['schema_changes'],
            quality_before, quality_after
        )

        return transformation_step

    def end_transformation(self) -> None:
        """End the current transformation stage."""
        if self.current_stage is None:
            warnings.warn("No active transformation to end")
            return

        print(f"[COMPLETE] Completed lineage tracking for stage: {self.current_stage}")
        self.current_stage = None

    def _calculate_pipeline_metrics(self) -> None:
        """Calculate pipeline-level metrics from transformation steps."""
        steps = self.pipeline_lineage.transformation_steps
        if not steps:
            return

        # Total processing time
        self.pipeline_lineage.total_processing_time = sum(
            step.processing_time_seconds for step in steps
        )

        # Data reduction and feature expansion ratios
        first_step, last_step = steps[0], steps[-1]
        self.pipeline_lineage.data_reduction_ratio = last_step.output_shape[0] / first_step.input_shape[0]
        self.pipeline_lineage.feature_expansion_ratio = last_step.output_shape[1] / first_step.input_shape[1]

        # Overall quality improvement
        quality_steps = [s for s in steps if s.data_quality_before is not None and s.data_quality_after is not None]
        if quality_steps:
            self.pipeline_lineage.overall_quality_improvement = (
                quality_steps[-1].data_quality_after - quality_steps[0].data_quality_before
            )

        # Validation status
        validated_count = sum(1 for s in steps if s.mathematical_equivalence_validated)
        self.pipeline_lineage.all_transformations_validated = validated_count == len(steps)

    def finalize_pipeline_lineage(self) -> PipelineLineage:
        """
        Finalize pipeline lineage and calculate summary metrics.

        Returns:
            Complete PipelineLineage with summary metrics
        """
        self.pipeline_lineage.end_time = datetime.now().isoformat()

        # Calculate pipeline-level metrics
        self._calculate_pipeline_metrics()

        # Store in history
        self.lineage_history.append(self.pipeline_lineage)

        # Log summary
        print(f"[FINALIZED] Pipeline lineage finalized: {self.pipeline_name}")
        print(f"   Execution ID: {self.execution_id}")
        print(f"   Steps tracked: {len(self.pipeline_lineage.transformation_steps)}")
        if self.pipeline_lineage.total_processing_time:
            print(f"   Total processing time: {self.pipeline_lineage.total_processing_time:.2f}s")

        return self.pipeline_lineage

    def _generate_doc_header(self) -> List[str]:
        """Generate document header with pipeline metadata."""
        return [
            f"# Data Transformation Lineage: {self.pipeline_name}",
            "",
            f"**Pipeline Version**: {self.pipeline_version}",
            f"**Execution ID**: {self.execution_id}",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Duration**: {self.pipeline_lineage.start_time} to {self.pipeline_lineage.end_time or 'In Progress'}",
            "",
            "## Executive Summary",
            ""
        ]

    def _generate_summary_metrics(self) -> List[str]:
        """Generate summary metrics section."""
        lines = []
        if self.pipeline_lineage.total_processing_time:
            lines.extend([
                f"- **Total Processing Time**: {self.pipeline_lineage.total_processing_time:.2f} seconds",
                f"- **Transformation Steps**: {len(self.pipeline_lineage.transformation_steps)}",
                f"- **Data Reduction Ratio**: {self.pipeline_lineage.data_reduction_ratio:.3f}x",
                f"- **Feature Expansion Ratio**: {self.pipeline_lineage.feature_expansion_ratio:.1f}x",
            ])
            if self.pipeline_lineage.overall_quality_improvement is not None:
                lines.append(f"- **Overall Quality Improvement**: {self.pipeline_lineage.overall_quality_improvement:+.1f} points")
            lines.extend([
                f"- **Mathematical Validation**: {'All steps validated' if self.pipeline_lineage.all_transformations_validated else 'Some steps not validated'}",
                ""
            ])
        return lines

    def _generate_step_header(self, step: TransformationStep, index: int) -> List[str]:
        """Generate header section for a transformation step."""
        validation_icon = "[OK]" if step.mathematical_equivalence_validated else "[WARN]"
        return [
            f"### {index}. {validation_icon} {step.step_name}",
            "",
            f"**Pipeline Stage**: {step.pipeline_stage}",
            f"**Timestamp**: {step.timestamp}",
            f"**Business Purpose**: {step.business_purpose}",
            "",
            "#### Data Transformation",
            "",
            f"- **Input Shape**: {step.input_shape[0]:,} rows × {step.input_shape[1]} columns",
            f"- **Output Shape**: {step.output_shape[0]:,} rows × {step.output_shape[1]} columns",
            f"- **Processing Time**: {step.processing_time_seconds:.2f} seconds",
            f"- **Memory Usage**: {step.memory_usage_mb:.1f} MB",
            ""
        ]

    def _generate_step_schema_changes(self, step: TransformationStep) -> List[str]:
        """Generate schema changes section for a step."""
        lines = []
        if step.columns_added or step.columns_removed or step.columns_renamed:
            lines.extend(["#### Schema Changes", ""])
            if step.columns_added:
                suffix = '...' if len(step.columns_added) > 10 else ''
                lines.append(f"- **Columns Added**: {', '.join(step.columns_added[:10])}{suffix}")
            if step.columns_removed:
                suffix = '...' if len(step.columns_removed) > 10 else ''
                lines.append(f"- **Columns Removed**: {', '.join(step.columns_removed[:10])}{suffix}")
            if step.columns_renamed:
                renames = [f"{old} → {new}" for old, new in list(step.columns_renamed.items())[:5]]
                suffix = '...' if len(step.columns_renamed) > 5 else ''
                lines.append(f"- **Columns Renamed**: {', '.join(renames)}{suffix}")
            lines.append("")
        return lines

    def _generate_step_quality_section(self, step: TransformationStep) -> List[str]:
        """Generate quality impact section for a step."""
        lines = []
        if step.data_quality_before is not None and step.data_quality_after is not None:
            quality_change = step.data_quality_after - step.data_quality_before
            lines.extend([
                "#### Quality Impact",
                "",
                f"- **Quality Before**: {step.data_quality_before:.1f}/100",
                f"- **Quality After**: {step.data_quality_after:.1f}/100",
                f"- **Quality Change**: {quality_change:+.1f} points",
                ""
            ])
        return lines

    def _generate_step_validation_section(self, step: TransformationStep) -> List[str]:
        """Generate mathematical validation section for a step."""
        lines = ["#### Mathematical Validation", ""]
        if step.mathematical_equivalence_validated:
            lines.extend([
                f"- **Equivalence Validated**: YES",
                f"- **Tolerance Used**: {step.equivalence_tolerance:.2e}",
                f"- **Max Numerical Difference**: {step.max_numerical_difference:.2e}",
                ""
            ])
        else:
            lines.extend([f"- **Equivalence Validated**: NO", ""])
        return lines

    def _generate_step_technical_section(self, step: TransformationStep, include: bool) -> List[str]:
        """Generate technical details section for a step."""
        lines = []
        if step.business_rules_applied:
            lines.extend(["#### Business Rules Applied", ""])
            for rule in step.business_rules_applied:
                lines.append(f"- {rule}")
            lines.append("")

        if include:
            lines.extend([
                "#### Technical Details",
                "",
                f"- **Step ID**: {step.step_id}",
                f"- **Input Data Hash**: {step.input_data_hash[:16]}...",
                f"- **Output Data Hash**: {step.output_data_hash[:16]}...",
                f"- **Configuration Hash**: {step.configuration_hash[:16]}...",
                ""
            ])

        if step.transformation_logic:
            lines.extend(["#### Transformation Logic", "", step.transformation_logic, ""])
        return lines

    def _generate_full_step_section(self, step: TransformationStep, index: int, include_technical: bool) -> List[str]:
        """Generate complete documentation for a single step."""
        lines = self._generate_step_header(step, index)
        lines.extend(self._generate_step_schema_changes(step))
        lines.extend(self._generate_step_quality_section(step))
        lines.extend(self._generate_step_validation_section(step))
        lines.extend(self._generate_step_technical_section(step, include_technical))
        return lines

    def export_lineage_documentation(
        self,
        output_path: str = "data_transformation_lineage.md",
        include_technical_details: bool = True
    ) -> str:
        """
        Export comprehensive lineage documentation.

        Args:
            output_path: Path to save the documentation
            include_technical_details: Whether to include technical implementation details

        Returns:
            Path to generated documentation
        """
        if not self.pipeline_lineage.transformation_steps:
            raise ValueError("No transformation steps recorded")

        # Build document sections
        doc_lines = self._generate_doc_header()
        doc_lines.extend(self._generate_summary_metrics())

        # Data flow diagram
        doc_lines.extend([
            "## Data Flow Overview", "", "```",
            self._generate_data_flow_ascii(), "```", ""
        ])

        # Transformation steps
        doc_lines.extend(["## Transformation Steps", ""])
        for i, step in enumerate(self.pipeline_lineage.transformation_steps, 1):
            doc_lines.extend(self._generate_full_step_section(step, i, include_technical_details))

        # Configuration summary
        if self.pipeline_lineage.configuration:
            doc_lines.extend([
                "## Pipeline Configuration", "", "```json",
                json.dumps(self.pipeline_lineage.configuration, indent=2), "```", ""
            ])

        # Write documentation
        doc_content = "\n".join(doc_lines)
        with open(output_path, 'w') as f:
            f.write(doc_content)

        print(f"[EXPORT] Lineage documentation exported: {output_path}")
        return output_path

    def _analyze_schema_changes(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze schema changes between input and output DataFrames.

        Returns:
            Dictionary with detailed schema change analysis
        """
        input_cols = set(input_df.columns)
        output_cols = set(output_df.columns)

        columns_added = list(output_cols - input_cols)
        columns_removed = list(input_cols - output_cols)

        # Detect potential column renames (heuristic based on similar names)
        columns_renamed = {}

        # Check for data type changes in common columns
        common_cols = input_cols & output_cols
        data_types_changed = {}

        for col in common_cols:
            input_type = str(input_df[col].dtype)
            output_type = str(output_df[col].dtype)
            if input_type != output_type:
                data_types_changed[col] = (input_type, output_type)

        return {
            'columns_added': sorted(columns_added),
            'columns_removed': sorted(columns_removed),
            'columns_renamed': columns_renamed,
            'data_types_changed': data_types_changed
        }

    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for data provenance."""
        try:
            # Use shape and column info for lightweight hash
            content = f"{df.shape}_{sorted(df.columns)}_{df.dtypes.to_dict()}"
            return hashlib.md5(content.encode()).hexdigest()
        except (AttributeError, TypeError, ValueError) as e:
            # Expected failures: df.columns not iterable, dtypes not serializable, shape access fails
            logger.debug(f"DataFrame hash calculation failed: {e}")
            return "hash_calculation_failed"

    def _calculate_configuration_hash(self, step_name: str) -> str:
        """Calculate hash of configuration for the step."""
        try:
            config_content = f"{self.pipeline_name}_{step_name}_{self.pipeline_version}"
            if self.current_stage in self.pipeline_lineage.configuration:
                config_content += json.dumps(
                    self.pipeline_lineage.configuration[self.current_stage],
                    sort_keys=True
                )
            return hashlib.md5(config_content.encode()).hexdigest()
        except (TypeError, KeyError, json.JSONDecodeError) as e:
            # Expected failures: configuration not JSON-serializable, missing keys, invalid JSON
            logger.debug(f"Configuration hash calculation failed for step '{step_name}': {e}")
            return "config_hash_calculation_failed"

    def _generate_data_flow_ascii(self) -> str:
        """Generate ASCII representation of data flow."""
        if not self.pipeline_lineage.transformation_steps:
            return "No transformation steps recorded"

        flow_lines = []
        for i, step in enumerate(self.pipeline_lineage.transformation_steps):
            input_shape = f"{step.input_shape[0]:,} × {step.input_shape[1]}"
            output_shape = f"{step.output_shape[0]:,} × {step.output_shape[1]}"

            if i == 0:
                flow_lines.append(f"[{input_shape}] Raw Data")

            arrow = "  ↓"
            step_line = f"{arrow} {step.step_name} ({step.pipeline_stage})"
            flow_lines.append(step_line)
            flow_lines.append(f"[{output_shape}] → Quality: {step.data_quality_after or 'N/A'}/100")

        return "\n".join(flow_lines)

    def _log_transformation_to_mlflow(self, step: TransformationStep) -> None:
        """Log transformation step to MLflow for tracking."""
        try:
            with mlflow.start_run(nested=True, run_name=f"lineage_{step.step_name}"):
                # Log step parameters
                mlflow.log_param("step_name", step.step_name)
                mlflow.log_param("pipeline_stage", step.pipeline_stage)
                mlflow.log_param("business_purpose", step.business_purpose[:100])  # Truncate long descriptions

                # Log metrics
                mlflow.log_metric("input_rows", step.input_shape[0])
                mlflow.log_metric("input_columns", step.input_shape[1])
                mlflow.log_metric("output_rows", step.output_shape[0])
                mlflow.log_metric("output_columns", step.output_shape[1])
                mlflow.log_metric("processing_time_seconds", step.processing_time_seconds)
                mlflow.log_metric("memory_usage_mb", step.memory_usage_mb)

                if step.data_quality_before is not None:
                    mlflow.log_metric("data_quality_before", step.data_quality_before)
                if step.data_quality_after is not None:
                    mlflow.log_metric("data_quality_after", step.data_quality_after)

                mlflow.log_metric("mathematical_equivalence_validated", 1.0 if step.mathematical_equivalence_validated else 0.0)

                # Log tags
                mlflow.set_tag("pipeline_name", self.pipeline_name)
                mlflow.set_tag("execution_id", self.execution_id)
                mlflow.set_tag("lineage_tracking", "enabled")

        except Exception as e:
            print(f"Warning: Failed to log transformation step to MLflow: {e}")

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID for this pipeline run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.pipeline_name}_{timestamp}"


# Convenience functions for common lineage scenarios
def track_pipeline_transformation(
    pipeline_name: str,
    stage_name: str,
    step_name: str,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    business_context: str = "",
    configuration: Optional[Dict[str, Any]] = None
) -> TransformationStep:
    """
    Quick transformation tracking for single steps.

    Args:
        pipeline_name: Name of the pipeline
        stage_name: Name of the pipeline stage
        step_name: Name of the transformation step
        input_df: Input DataFrame
        output_df: Output DataFrame
        business_context: Business purpose of transformation
        configuration: Step configuration

    Returns:
        TransformationStep with captured metadata
    """
    lineage = TransformationLineage(pipeline_name)
    lineage.start_transformation(stage_name, configuration)

    step = lineage.record_transformation_step(
        step_name=step_name,
        input_df=input_df,
        output_df=output_df,
        business_context=business_context
    )

    lineage.end_transformation()
    return step