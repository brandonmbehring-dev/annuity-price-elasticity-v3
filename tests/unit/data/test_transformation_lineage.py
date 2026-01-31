"""
Unit tests for src/data/transformation_lineage.py

Tests TransformationStep dataclass, PipelineLineage dataclass,
and TransformationLineage tracker class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import asdict


@pytest.fixture
def sample_input_df():
    """Sample input DataFrame for transformation testing."""
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
    })


@pytest.fixture
def sample_output_df():
    """Sample output DataFrame after transformation."""
    return pd.DataFrame({
        'col1': [2, 4, 6, 8, 10],  # Doubled
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [True, True, True, True, True],  # New column
    })


class TestTransformationStep:
    """Tests for TransformationStep dataclass."""

    def test_dataclass_creation(self):
        """Creates TransformationStep with all fields."""
        from src.data.transformation_lineage import TransformationStep

        step = TransformationStep(
            step_name='test_step',
            step_id='step_001',
            timestamp='2024-01-01T00:00:00',
            pipeline_stage='preprocessing',
            input_shape=(100, 5),
            output_shape=(95, 6),
            input_columns=['a', 'b', 'c', 'd', 'e'],
            output_columns=['a', 'b', 'c', 'd', 'e', 'f'],
            columns_added=['f'],
            columns_removed=[],
            columns_renamed={},
            data_types_changed={},
            business_purpose='Filter and add new feature',
            transformation_logic='Filter by condition, add computed column',
            business_rules_applied=['rule_1', 'rule_2'],
            data_quality_before=95.0,
            data_quality_after=97.0,
            processing_time_seconds=1.5,
            memory_usage_mb=50.0,
            mathematical_equivalence_validated=True,
            equivalence_tolerance=1e-12,
            max_numerical_difference=1e-15,
            input_data_hash='abc123',
            output_data_hash='def456',
            configuration_hash='config789'
        )
        assert step.step_name == 'test_step'
        assert step.input_shape == (100, 5)
        assert step.columns_added == ['f']

    def test_dataclass_conversion_to_dict(self):
        """Converts to dict via asdict."""
        from src.data.transformation_lineage import TransformationStep

        step = TransformationStep(
            step_name='test',
            step_id='s1',
            timestamp='2024-01-01',
            pipeline_stage='test',
            input_shape=(10, 2),
            output_shape=(10, 3),
            input_columns=['a', 'b'],
            output_columns=['a', 'b', 'c'],
            columns_added=['c'],
            columns_removed=[],
            columns_renamed={},
            data_types_changed={},
            business_purpose='test',
            transformation_logic='test',
            business_rules_applied=[],
            data_quality_before=None,
            data_quality_after=None,
            processing_time_seconds=0.1,
            memory_usage_mb=1.0,
            mathematical_equivalence_validated=False,
            equivalence_tolerance=None,
            max_numerical_difference=None,
            input_data_hash='hash1',
            output_data_hash='hash2',
            configuration_hash='hash3'
        )
        d = asdict(step)
        assert isinstance(d, dict)
        assert d['step_name'] == 'test'


class TestPipelineLineage:
    """Tests for PipelineLineage dataclass."""

    def test_dataclass_creation(self):
        """Creates PipelineLineage with required fields."""
        from src.data.transformation_lineage import PipelineLineage

        lineage = PipelineLineage(
            pipeline_name='test_pipeline',
            pipeline_version='1.0',
            execution_id='exec_001',
            start_time='2024-01-01T00:00:00',
            end_time=None,
            configuration={'param1': 'value1'},
            business_parameters={'product': '6Y20B'}
        )
        assert lineage.pipeline_name == 'test_pipeline'
        assert lineage.transformation_steps == []

    def test_default_values(self):
        """Default values are set correctly."""
        from src.data.transformation_lineage import PipelineLineage

        lineage = PipelineLineage(
            pipeline_name='test',
            pipeline_version='1.0',
            execution_id='e1',
            start_time='2024-01-01',
            end_time=None,
            configuration={},
            business_parameters={}
        )
        assert lineage.transformation_steps == []
        assert lineage.total_processing_time is None
        assert lineage.all_transformations_validated is False


class TestTransformationLineage:
    """Tests for TransformationLineage class."""

    def test_initialization(self):
        """Initializes with pipeline name and version."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test_pipeline', '2.0')
        assert lineage.pipeline_name == 'test_pipeline'
        assert lineage.pipeline_version == '2.0'

    def test_default_version(self):
        """Default version is 1.0."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test_pipeline')
        assert lineage.pipeline_version == '1.0'

    def test_generates_execution_id(self):
        """Generates execution ID containing pipeline name."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        assert 'test' in lineage.execution_id


class TestTransformationTracking:
    """Tests for transformation tracking methods."""

    def test_start_transformation(self):
        """Starts transformation recording."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test_pipeline')
        lineage.start_transformation('filtering_stage')
        assert lineage.current_stage == 'filtering_stage'

    def test_end_transformation(self):
        """Ends transformation recording."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test_pipeline')
        lineage.start_transformation('filtering_stage')
        lineage.end_transformation()
        assert lineage.current_stage is None

    def test_record_transformation_step(self, sample_input_df, sample_output_df):
        """Records transformation step metadata."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test_pipeline')
        lineage.start_transformation('preprocessing')

        lineage.record_transformation_step(
            step_name='double_values',
            input_df=sample_input_df,
            output_df=sample_output_df,
            business_context='Double col1 values for analysis'
        )

        lineage.end_transformation()

        # Check step was recorded
        assert len(lineage.pipeline_lineage.transformation_steps) == 1
        step = lineage.pipeline_lineage.transformation_steps[0]
        assert step.step_name == 'double_values'


class TestDataHashing:
    """Tests for data hashing functionality."""

    def test_hash_dataframe(self, sample_input_df):
        """Generates consistent hash for DataFrame."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        hash1 = lineage._calculate_dataframe_hash(sample_input_df)
        hash2 = lineage._calculate_dataframe_hash(sample_input_df)
        assert hash1 == hash2

    def test_different_data_different_hash(self, sample_input_df, sample_output_df):
        """Different DataFrames have different hashes."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        hash1 = lineage._calculate_dataframe_hash(sample_input_df)
        hash2 = lineage._calculate_dataframe_hash(sample_output_df)
        assert hash1 != hash2


class TestSchemaChangeDetection:
    """Tests for schema change detection."""

    def test_detect_columns_added(self, sample_input_df, sample_output_df):
        """Detects added columns between input and output."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        changes = lineage._analyze_schema_changes(sample_input_df, sample_output_df)
        assert 'col3' in changes['columns_added']

    def test_detect_columns_removed(self):
        """Detects removed columns between input and output."""
        from src.data.transformation_lineage import TransformationLineage

        input_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        output_df = pd.DataFrame({'a': [1], 'b': [2]})

        lineage = TransformationLineage('test')
        changes = lineage._analyze_schema_changes(input_df, output_df)
        assert 'c' in changes['columns_removed']

    def test_no_changes_detected(self, sample_input_df):
        """No changes detected when schemas match."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        changes = lineage._analyze_schema_changes(sample_input_df, sample_input_df)
        assert changes['columns_added'] == []
        assert changes['columns_removed'] == []


class TestLineageExport:
    """Tests for lineage export functionality."""

    def test_export_lineage_documentation(self, sample_input_df, sample_output_df, tmp_path):
        """Exports lineage to markdown file."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test_pipeline')
        lineage.start_transformation('stage1')
        lineage.record_transformation_step(
            step_name='step1',
            input_df=sample_input_df,
            output_df=sample_output_df,
            business_context='Test transformation'
        )
        lineage.end_transformation()

        output_path = tmp_path / 'lineage.md'
        lineage.export_lineage_documentation(str(output_path))
        assert output_path.exists()

    def test_export_requires_transformation_steps(self):
        """Export raises error without transformation steps."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test_pipeline')
        with pytest.raises(ValueError, match="No transformation steps recorded"):
            lineage.export_lineage_documentation('output.md')


class TestImportChecks:
    """Tests for optional import handling."""

    def test_mlflow_availability_flag_exists(self):
        """MLFLOW_AVAILABLE flag is defined."""
        from src.data.transformation_lineage import MLFLOW_AVAILABLE
        assert isinstance(MLFLOW_AVAILABLE, bool)

    def test_graphviz_availability_flag_exists(self):
        """GRAPHVIZ_AVAILABLE flag is defined."""
        from src.data.transformation_lineage import GRAPHVIZ_AVAILABLE
        assert isinstance(GRAPHVIZ_AVAILABLE, bool)


class TestPipelineFinalization:
    """Tests for pipeline finalization."""

    def test_finalize_sets_end_time(self, sample_input_df, sample_output_df):
        """Finalization sets end time."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test_pipeline')
        lineage.start_transformation('stage1')
        lineage.record_transformation_step(
            step_name='step1',
            input_df=sample_input_df,
            output_df=sample_output_df
        )
        lineage.end_transformation()

        result = lineage.finalize_pipeline_lineage()
        assert result.end_time is not None

    def test_finalize_returns_pipeline_lineage(self, sample_input_df, sample_output_df):
        """Finalization returns PipelineLineage."""
        from src.data.transformation_lineage import TransformationLineage, PipelineLineage

        lineage = TransformationLineage('test_pipeline')
        lineage.start_transformation('stage1')
        lineage.record_transformation_step(
            step_name='step1',
            input_df=sample_input_df,
            output_df=sample_output_df
        )
        lineage.end_transformation()

        result = lineage.finalize_pipeline_lineage()
        assert isinstance(result, PipelineLineage)

    def test_finalize_calculates_data_reduction_ratio(self):
        """Calculates data reduction ratio."""
        from src.data.transformation_lineage import TransformationLineage

        input_df = pd.DataFrame({'a': range(100)})
        output_df = pd.DataFrame({'a': range(50)})

        lineage = TransformationLineage('test')
        lineage.start_transformation('filter')
        lineage.record_transformation_step(
            step_name='filter_step',
            input_df=input_df,
            output_df=output_df,
            business_context='Filter to 50%'
        )
        lineage.end_transformation()

        step = lineage.pipeline_lineage.transformation_steps[0]
        assert step.input_shape[0] == 100
        assert step.output_shape[0] == 50


class TestRecordTransformationStep:
    """Tests for record_transformation_step method edge cases."""

    def test_record_requires_start_transformation(self, sample_input_df, sample_output_df):
        """Recording without starting transformation raises ValueError."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')

        with pytest.raises(ValueError, match="Must call start_transformation"):
            lineage.record_transformation_step(
                step_name='test',
                input_df=sample_input_df,
                output_df=sample_output_df
            )

    def test_record_with_quality_metrics(self, sample_input_df, sample_output_df):
        """Records quality metrics when provided."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        lineage.start_transformation('stage1')

        step = lineage.record_transformation_step(
            step_name='quality_step',
            input_df=sample_input_df,
            output_df=sample_output_df,
            quality_before=85.0,
            quality_after=92.0
        )

        lineage.end_transformation()

        assert step.data_quality_before == 85.0
        assert step.data_quality_after == 92.0

    def test_record_with_mathematical_validation(self, sample_input_df, sample_output_df):
        """Records mathematical validation parameters."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        lineage.start_transformation('stage1')

        step = lineage.record_transformation_step(
            step_name='validated_step',
            input_df=sample_input_df,
            output_df=sample_output_df,
            equivalence_validated=True,
            equivalence_tolerance=1e-12,
            max_numerical_diff=1e-15
        )

        lineage.end_transformation()

        assert step.mathematical_equivalence_validated is True
        assert step.equivalence_tolerance == 1e-12
        assert step.max_numerical_difference == 1e-15


class TestDataFlowAscii:
    """Tests for ASCII data flow representation."""

    def test_generate_empty_flow(self):
        """Handles empty transformation steps."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        flow = lineage._generate_data_flow_ascii()

        assert 'No transformation steps recorded' in flow

    def test_generate_flow_with_steps(self, sample_input_df, sample_output_df):
        """Generates ASCII flow diagram with steps."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        lineage.start_transformation('preprocessing')

        lineage.record_transformation_step(
            step_name='transform_data',
            input_df=sample_input_df,
            output_df=sample_output_df,
            quality_after=95.0
        )

        lineage.end_transformation()

        flow = lineage._generate_data_flow_ascii()

        assert 'Raw Data' in flow
        assert 'transform_data' in flow
        assert '↓' in flow


class TestConfigurationHash:
    """Tests for configuration hash calculation."""

    def test_calculate_configuration_hash(self):
        """Calculates consistent configuration hash."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        lineage.start_transformation('stage1', configuration={'param': 'value'})

        hash1 = lineage._calculate_configuration_hash('test_step')
        hash2 = lineage._calculate_configuration_hash('test_step')

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_configuration_hash_fallback_on_error(self):
        """Returns fallback hash on serialization error."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        lineage.start_transformation('stage1')
        # Add non-serializable configuration
        lineage.pipeline_lineage.configuration['stage1'] = {'func': lambda x: x}

        hash_result = lineage._calculate_configuration_hash('test_step')

        assert hash_result == 'config_hash_calculation_failed'


class TestDataFrameHash:
    """Tests for DataFrame hash edge cases."""

    def test_hash_handles_empty_dataframe(self):
        """Handles empty DataFrame for hashing."""
        from src.data.transformation_lineage import TransformationLineage

        lineage = TransformationLineage('test')
        df = pd.DataFrame()

        hash_result = lineage._calculate_dataframe_hash(df)

        # Should return a valid hash (not fail)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32


class TestConvenienceFunction:
    """Tests for module-level convenience function."""

    def test_track_pipeline_transformation(self, sample_input_df, sample_output_df):
        """track_pipeline_transformation convenience function."""
        from src.data.transformation_lineage import track_pipeline_transformation

        step = track_pipeline_transformation(
            pipeline_name='test_pipeline',
            stage_name='test_stage',
            step_name='test_step',
            input_df=sample_input_df,
            output_df=sample_output_df,
            business_context='Test transformation',
            configuration={'param': 'value'}
        )

        assert step.step_name == 'test_step'
        assert step.pipeline_stage == 'test_stage'
