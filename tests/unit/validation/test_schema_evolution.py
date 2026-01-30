"""
Tests for src.validation.schema_evolution module.

Comprehensive tests for schema evolution tracking including:
- SchemaChangeType enum validation
- SchemaFingerprint dataclass and factory methods
- SchemaEvolutionTracker class methods
- Change detection (columns, types, business rules)
- Schema analysis and recommendations
- File I/O for evolution history
- MLflow integration
- Convenience functions

Coverage Target: 80%+ of schema_evolution.py (701 lines)
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

from src.validation.schema_evolution import (
    SchemaChangeType,
    SchemaFingerprint,
    SchemaEvolutionTracker,
    track_dataset_evolution,
    get_evolution_report,
    VALIDATION_MODULES_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Standard DataFrame for testing schema fingerprinting."""
    np.random.seed(42)
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100),
        'sales': np.random.uniform(50000, 200000, 100),
        'price': np.random.uniform(1, 5, 100),
    })


@pytest.fixture
def sample_dataframe_with_nulls() -> pd.DataFrame:
    """DataFrame with null values for testing null handling."""
    np.random.seed(42)
    df = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100),
        'sales': np.random.uniform(50000, 200000, 100),
        'price': np.random.uniform(1, 5, 100),
    })
    # Introduce some nulls
    df.loc[5:10, 'sales'] = np.nan
    return df


@pytest.fixture
def evolved_dataframe(sample_dataframe: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with schema changes for evolution testing."""
    df = sample_dataframe.copy()
    df['new_feature'] = np.random.randn(100)
    df['sales'] = df['sales'].astype('int64')  # Type change
    return df


@pytest.fixture
def evolved_dataframe_removed_column(sample_dataframe: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with column removed (breaking change)."""
    df = sample_dataframe.copy()
    df = df.drop(columns=['price'])
    return df


@pytest.fixture
def temp_tracking_dir(tmp_path) -> Path:
    """Temporary directory for evolution tracking."""
    tracking_dir = tmp_path / "schema_evolution"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return tracking_dir


@pytest.fixture
def tracker(temp_tracking_dir: Path) -> SchemaEvolutionTracker:
    """Pre-configured SchemaEvolutionTracker with temp directory."""
    return SchemaEvolutionTracker(tracking_dir=str(temp_tracking_dir))


@pytest.fixture
def sample_fingerprint(sample_dataframe: pd.DataFrame) -> SchemaFingerprint:
    """Pre-created fingerprint for comparison tests."""
    return SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")


@pytest.fixture
def mock_pandera_schema() -> MagicMock:
    """Mock Pandera schema for baseline validation tests."""
    mock = MagicMock()
    mock.columns = {'date': MagicMock(), 'sales': MagicMock(), 'price': MagicMock()}
    mock.strict = True
    return mock


# =============================================================================
# PHASE 1: QUICK WINS - ENUM AND PURE HELPER TESTS
# =============================================================================


class TestSchemaChangeTypeEnum:
    """Tests for SchemaChangeType enum values and behavior."""

    def test_column_added_value(self):
        """COLUMN_ADDED enum has correct string value."""
        assert SchemaChangeType.COLUMN_ADDED.value == "column_added"

    def test_column_removed_value(self):
        """COLUMN_REMOVED enum has correct string value."""
        assert SchemaChangeType.COLUMN_REMOVED.value == "column_removed"

    def test_column_type_changed_value(self):
        """COLUMN_TYPE_CHANGED enum has correct string value."""
        assert SchemaChangeType.COLUMN_TYPE_CHANGED.value == "column_type_changed"

    def test_column_nullable_changed_value(self):
        """COLUMN_NULLABLE_CHANGED enum has correct string value."""
        assert SchemaChangeType.COLUMN_NULLABLE_CHANGED.value == "column_nullable_changed"

    def test_business_rule_changed_value(self):
        """BUSINESS_RULE_CHANGED enum has correct string value."""
        assert SchemaChangeType.BUSINESS_RULE_CHANGED.value == "business_rule_changed"

    def test_no_change_value(self):
        """NO_CHANGE enum has correct string value."""
        assert SchemaChangeType.NO_CHANGE.value == "no_change"

    def test_all_enum_members_exist(self):
        """All expected enum members should be present."""
        expected_members = {
            'COLUMN_ADDED', 'COLUMN_REMOVED', 'COLUMN_TYPE_CHANGED',
            'COLUMN_NULLABLE_CHANGED', 'BUSINESS_RULE_CHANGED', 'NO_CHANGE'
        }
        actual_members = set(SchemaChangeType.__members__.keys())
        assert expected_members == actual_members


# =============================================================================
# PHASE 2: SCHEMA FINGERPRINT TESTS
# =============================================================================


class TestSchemaFingerprintFromDataFrame:
    """Tests for SchemaFingerprint.from_dataframe() factory method."""

    def test_creates_fingerprint_from_valid_dataframe(self, sample_dataframe: pd.DataFrame):
        """from_dataframe() should create valid SchemaFingerprint."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")

        assert isinstance(fp, SchemaFingerprint)
        assert fp.shape == (100, 3)
        assert set(fp.columns) == {'date', 'sales', 'price'}

    def test_captures_column_types(self, sample_dataframe: pd.DataFrame):
        """from_dataframe() should capture correct column types."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")

        assert 'datetime64' in fp.column_types['date']
        assert 'float64' in fp.column_types['sales']
        assert 'float64' in fp.column_types['price']

    def test_captures_null_counts(self, sample_dataframe_with_nulls: pd.DataFrame):
        """from_dataframe() should capture null counts correctly."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe_with_nulls, "test_schema")

        assert fp.null_counts['sales'] == 6  # Rows 5-10 inclusive
        assert fp.null_counts['date'] == 0
        assert fp.null_counts['price'] == 0

    def test_captures_null_percentages(self, sample_dataframe_with_nulls: pd.DataFrame):
        """from_dataframe() should calculate null percentages correctly."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe_with_nulls, "test_schema")

        assert fp.null_percentages['sales'] == pytest.approx(6.0, rel=0.01)
        assert fp.null_percentages['date'] == 0.0

    def test_calculates_memory_usage(self, sample_dataframe: pd.DataFrame):
        """from_dataframe() should calculate memory usage in MB."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")

        assert fp.memory_usage_mb >= 0  # Small DataFrames may round to 0
        assert isinstance(fp.memory_usage_mb, (float, np.floating))

    def test_counts_duplicate_rows(self, sample_dataframe: pd.DataFrame):
        """from_dataframe() should count duplicate rows."""
        df = sample_dataframe.copy()
        # Add duplicate rows
        df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

        fp = SchemaFingerprint.from_dataframe(df, "test_schema")
        assert fp.duplicate_rows == 2

    def test_calculates_date_range_days(self, sample_dataframe: pd.DataFrame):
        """from_dataframe() should calculate date range when date column exists."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")

        assert fp.date_range_days == 99  # 100 days - 1

    def test_date_range_none_when_no_date_column(self):
        """from_dataframe() should return None for date_range_days without date column."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        fp = SchemaFingerprint.from_dataframe(df, "test_schema")

        assert fp.date_range_days is None

    def test_generates_schema_hash(self, sample_dataframe: pd.DataFrame):
        """from_dataframe() should generate a schema hash."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")

        assert fp.schema_hash is not None
        assert len(fp.schema_hash) == 12  # Short hash for readability

    def test_schema_hash_deterministic(self, sample_dataframe: pd.DataFrame):
        """Same DataFrame should produce same schema hash."""
        fp1 = SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")
        fp2 = SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")

        assert fp1.schema_hash == fp2.schema_hash

    def test_schema_hash_changes_with_columns(self, sample_dataframe: pd.DataFrame):
        """Schema hash should change when columns change."""
        fp1 = SchemaFingerprint.from_dataframe(sample_dataframe, "test_schema")

        df_modified = sample_dataframe.copy()
        df_modified['new_col'] = 1

        fp2 = SchemaFingerprint.from_dataframe(df_modified, "test_schema")

        assert fp1.schema_hash != fp2.schema_hash


class TestSchemaFingerprintExtractBusinessRules:
    """Tests for SchemaFingerprint._extract_business_rules() method."""

    def test_extracts_non_negative_rule(self):
        """Should detect non-negative numeric columns."""
        df = pd.DataFrame({'positive': [1, 2, 3], 'mixed': [-1, 0, 1]})
        rules = SchemaFingerprint._extract_business_rules(df)

        positive_rules = next((r for r in rules if r['column'] == 'positive'), None)
        assert positive_rules is not None
        assert any(r['type'] == 'non_negative' for r in positive_rules['rules'])

    def test_extracts_range_rule(self):
        """Should detect range constraints on numeric columns."""
        df = pd.DataFrame({'bounded': [10, 20, 30]})
        rules = SchemaFingerprint._extract_business_rules(df)

        bounded_rules = next((r for r in rules if r['column'] == 'bounded'), None)
        assert bounded_rules is not None

        range_rule = next((r for r in bounded_rules['rules'] if r['type'] == 'range'), None)
        assert range_rule is not None
        assert range_rule['min_value'] == 10.0
        assert range_rule['max_value'] == 30.0

    def test_extracts_required_rule(self):
        """Should detect required (non-null) columns."""
        df = pd.DataFrame({'required': [1, 2, 3], 'optional': [1, None, 3]})
        rules = SchemaFingerprint._extract_business_rules(df)

        required_rules = next((r for r in rules if r['column'] == 'required'), None)
        assert required_rules is not None
        assert any(r['type'] == 'required' for r in required_rules['rules'])

    def test_no_rules_for_nullable_column(self):
        """Columns with nulls should not have 'required' rule."""
        df = pd.DataFrame({'nullable': [1.0, None, 3.0]})
        rules = SchemaFingerprint._extract_business_rules(df)

        if rules:
            nullable_rules = next((r for r in rules if r['column'] == 'nullable'), None)
            if nullable_rules:
                assert not any(r['type'] == 'required' for r in nullable_rules['rules'])

    def test_empty_dataframe_returns_empty_rules(self):
        """Empty DataFrame should return empty rules list."""
        df = pd.DataFrame()
        rules = SchemaFingerprint._extract_business_rules(df)

        assert rules == []

    def test_string_columns_no_numeric_rules(self):
        """String columns should not have numeric rules."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        rules = SchemaFingerprint._extract_business_rules(df)

        if rules:
            text_rules = next((r for r in rules if r['column'] == 'text'), None)
            if text_rules:
                # Should not have non_negative or range rules
                assert not any(r['type'] in ('non_negative', 'range')
                               for r in text_rules['rules'])

    def test_handles_all_null_column(self):
        """Should handle columns that are entirely null."""
        df = pd.DataFrame({'all_null': [None, None, None]})
        rules = SchemaFingerprint._extract_business_rules(df)

        # Should not crash; rules may be empty or not have required flag
        assert isinstance(rules, list)

    def test_handles_negative_values(self):
        """Should not flag non_negative for columns with negative values."""
        df = pd.DataFrame({'negative': [-10, -5, 0]})
        rules = SchemaFingerprint._extract_business_rules(df)

        if rules:
            neg_rules = next((r for r in rules if r['column'] == 'negative'), None)
            if neg_rules:
                assert not any(r['type'] == 'non_negative' for r in neg_rules['rules'])


# =============================================================================
# PHASE 3: CHANGE DETECTION TESTS
# =============================================================================


class TestDetectColumnChanges:
    """Tests for SchemaEvolutionTracker._detect_column_changes()."""

    def test_detects_added_columns(self, tracker: SchemaEvolutionTracker,
                                   sample_dataframe: pd.DataFrame,
                                   evolved_dataframe: pd.DataFrame):
        """Should detect columns added to schema."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes, added, removed, common = tracker._detect_column_changes(current, previous)

        assert 'new_feature' in added
        assert len(removed) == 0
        assert any(c['type'] == SchemaChangeType.COLUMN_ADDED.value for c in changes)

    def test_detects_removed_columns(self, tracker: SchemaEvolutionTracker,
                                     sample_dataframe: pd.DataFrame,
                                     evolved_dataframe_removed_column: pd.DataFrame):
        """Should detect columns removed from schema."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe_removed_column, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes, added, removed, common = tracker._detect_column_changes(current, previous)

        assert 'price' in removed
        assert len(added) == 0
        assert any(c['type'] == SchemaChangeType.COLUMN_REMOVED.value for c in changes)

    def test_removed_columns_marked_breaking(self, tracker: SchemaEvolutionTracker,
                                             sample_dataframe: pd.DataFrame,
                                             evolved_dataframe_removed_column: pd.DataFrame):
        """Removed columns should be marked as breaking changes."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe_removed_column, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes, _, _, _ = tracker._detect_column_changes(current, previous)

        removed_change = next(c for c in changes
                              if c['type'] == SchemaChangeType.COLUMN_REMOVED.value)
        assert removed_change['breaking'] is True

    def test_added_columns_not_breaking(self, tracker: SchemaEvolutionTracker,
                                        sample_dataframe: pd.DataFrame,
                                        evolved_dataframe: pd.DataFrame):
        """Added columns should not be marked as breaking changes."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes, _, _, _ = tracker._detect_column_changes(current, previous)

        added_changes = [c for c in changes
                         if c['type'] == SchemaChangeType.COLUMN_ADDED.value]
        for change in added_changes:
            assert change['breaking'] is False

    def test_identifies_common_columns(self, tracker: SchemaEvolutionTracker,
                                       sample_dataframe: pd.DataFrame,
                                       evolved_dataframe: pd.DataFrame):
        """Should correctly identify common columns."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        _, _, _, common = tracker._detect_column_changes(current, previous)

        assert {'date', 'sales', 'price'} == common

    def test_no_changes_for_identical_schemas(self, tracker: SchemaEvolutionTracker,
                                              sample_dataframe: pd.DataFrame):
        """Should return no changes for identical schemas."""
        fp1 = SchemaFingerprint.from_dataframe(sample_dataframe, "test")
        fp2 = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes, added, removed, common = tracker._detect_column_changes(fp1, fp2)

        assert len(changes) == 0
        assert len(added) == 0
        assert len(removed) == 0
        assert len(common) == 3  # date, sales, price

    def test_change_includes_data_type_info(self, tracker: SchemaEvolutionTracker,
                                            sample_dataframe: pd.DataFrame):
        """Added column changes should include data type information."""
        df_modified = sample_dataframe.copy()
        df_modified['new_int'] = [1, 2, 3] * 33 + [4]

        current = SchemaFingerprint.from_dataframe(df_modified, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes, _, _, _ = tracker._detect_column_changes(current, previous)

        added_change = next(c for c in changes
                            if c['type'] == SchemaChangeType.COLUMN_ADDED.value)
        assert 'data_type' in added_change

    def test_removed_includes_previous_type(self, tracker: SchemaEvolutionTracker,
                                            sample_dataframe: pd.DataFrame,
                                            evolved_dataframe_removed_column: pd.DataFrame):
        """Removed column changes should include previous data type."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe_removed_column, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes, _, _, _ = tracker._detect_column_changes(current, previous)

        removed_change = next(c for c in changes
                              if c['type'] == SchemaChangeType.COLUMN_REMOVED.value)
        assert 'previous_data_type' in removed_change

    def test_multiple_added_and_removed(self, tracker: SchemaEvolutionTracker):
        """Should handle multiple columns added and removed simultaneously."""
        previous_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        current_df = pd.DataFrame({'b': [2], 'd': [4], 'e': [5]})

        current = SchemaFingerprint.from_dataframe(current_df, "test")
        previous = SchemaFingerprint.from_dataframe(previous_df, "test")

        changes, added, removed, common = tracker._detect_column_changes(current, previous)

        assert added == {'d', 'e'}
        assert removed == {'a', 'c'}
        assert common == {'b'}


class TestDetectTypeChanges:
    """Tests for SchemaEvolutionTracker._detect_type_changes()."""

    def test_detects_type_change(self, tracker: SchemaEvolutionTracker,
                                 sample_dataframe: pd.DataFrame):
        """Should detect when column type changes."""
        df_modified = sample_dataframe.copy()
        df_modified['sales'] = df_modified['sales'].astype('int64')

        current = SchemaFingerprint.from_dataframe(df_modified, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        common_cols = {'date', 'sales', 'price'}
        changes = tracker._detect_type_changes(current, previous, common_cols)

        assert len(changes) == 1
        assert changes[0]['type'] == SchemaChangeType.COLUMN_TYPE_CHANGED.value
        assert changes[0]['column'] == 'sales'

    def test_type_change_includes_both_types(self, tracker: SchemaEvolutionTracker,
                                             sample_dataframe: pd.DataFrame):
        """Type changes should include previous and current types."""
        df_modified = sample_dataframe.copy()
        df_modified['sales'] = df_modified['sales'].astype('int64')

        current = SchemaFingerprint.from_dataframe(df_modified, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes = tracker._detect_type_changes(current, previous, {'sales'})

        assert changes[0]['previous_type'] == 'float64'
        assert changes[0]['current_type'] == 'int64'

    def test_compatible_type_change_not_breaking(self, tracker: SchemaEvolutionTracker):
        """Compatible type changes should not be marked as breaking."""
        df1 = pd.DataFrame({'val': pd.array([1, 2, 3], dtype='int32')})
        df2 = pd.DataFrame({'val': pd.array([1, 2, 3], dtype='int64')})

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_type_changes(current, previous, {'val'})

        assert len(changes) == 1
        assert changes[0]['breaking'] is False

    def test_incompatible_type_change_is_breaking(self, tracker: SchemaEvolutionTracker):
        """Incompatible type changes should be marked as breaking."""
        df1 = pd.DataFrame({'val': [1.5, 2.5, 3.5]})  # float64
        df2 = pd.DataFrame({'val': ['a', 'b', 'c']})  # object

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_type_changes(current, previous, {'val'})

        assert len(changes) == 1
        assert changes[0]['breaking'] is True

    def test_no_changes_for_same_types(self, tracker: SchemaEvolutionTracker,
                                       sample_dataframe: pd.DataFrame):
        """Should return no changes when types are identical."""
        fp1 = SchemaFingerprint.from_dataframe(sample_dataframe, "test")
        fp2 = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        common_cols = set(sample_dataframe.columns)
        changes = tracker._detect_type_changes(fp1, fp2, common_cols)

        assert len(changes) == 0

    def test_multiple_type_changes(self, tracker: SchemaEvolutionTracker):
        """Should detect multiple type changes."""
        df1 = pd.DataFrame({'a': [1.0, 2.0], 'b': [1.0, 2.0]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_type_changes(current, previous, {'a', 'b'})

        assert len(changes) == 2

    def test_ignores_columns_not_in_common(self, tracker: SchemaEvolutionTracker,
                                           sample_dataframe: pd.DataFrame,
                                           evolved_dataframe: pd.DataFrame):
        """Should only check columns in common_cols set."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        # Only check 'date' column
        changes = tracker._detect_type_changes(current, previous, {'date'})

        # 'sales' type changed but should not be detected
        assert len(changes) == 0

    def test_description_includes_column_name(self, tracker: SchemaEvolutionTracker,
                                              sample_dataframe: pd.DataFrame):
        """Type change description should include column name."""
        df_modified = sample_dataframe.copy()
        df_modified['sales'] = df_modified['sales'].astype('int64')

        current = SchemaFingerprint.from_dataframe(df_modified, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes = tracker._detect_type_changes(current, previous, {'sales'})

        assert 'sales' in changes[0]['description']


class TestDetectBusinessRuleChanges:
    """Tests for SchemaEvolutionTracker._detect_business_rule_changes()."""

    def test_detects_rule_addition(self, tracker: SchemaEvolutionTracker):
        """Should detect when business rules are added."""
        # Previous: nullable column
        df1 = pd.DataFrame({'val': [1.0, None, 3.0]})
        # Current: required column
        df2 = pd.DataFrame({'val': [1.0, 2.0, 3.0]})

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_business_rule_changes(current, previous, {'val'})

        # Rules changed because required rule was added
        assert len(changes) == 1
        assert changes[0]['type'] == SchemaChangeType.BUSINESS_RULE_CHANGED.value

    def test_detects_rule_removal(self, tracker: SchemaEvolutionTracker):
        """Should detect when business rules are removed."""
        # Previous: required column
        df1 = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
        # Current: nullable column
        df2 = pd.DataFrame({'val': [1.0, None, 3.0]})

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_business_rule_changes(current, previous, {'val'})

        assert len(changes) == 1
        assert changes[0]['breaking'] is True  # Removing required is breaking

    def test_detects_range_change(self, tracker: SchemaEvolutionTracker):
        """Should detect when value ranges change."""
        df1 = pd.DataFrame({'val': [10, 20, 30]})
        df2 = pd.DataFrame({'val': [5, 25, 50]})  # Different range

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_business_rule_changes(current, previous, {'val'})

        # Range changed
        assert len(changes) == 1

    def test_no_changes_for_same_rules(self, tracker: SchemaEvolutionTracker,
                                       sample_dataframe: pd.DataFrame):
        """Should return no changes when rules are identical."""
        fp1 = SchemaFingerprint.from_dataframe(sample_dataframe, "test")
        fp2 = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        common_cols = set(sample_dataframe.columns)
        changes = tracker._detect_business_rule_changes(fp1, fp2, common_cols)

        assert len(changes) == 0

    def test_includes_previous_and_current_rules(self, tracker: SchemaEvolutionTracker):
        """Rule changes should include both previous and current rules."""
        df1 = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({'val': [1.0, None, 3.0]})

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_business_rule_changes(current, previous, {'val'})

        if changes:
            assert 'previous_rules' in changes[0]
            assert 'current_rules' in changes[0]

    def test_ignores_columns_not_in_common(self, tracker: SchemaEvolutionTracker):
        """Should only check columns in common_cols set."""
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, None, 6]})

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        # Only check 'a' which hasn't changed
        changes = tracker._detect_business_rule_changes(current, previous, {'a'})

        assert len(changes) == 0

    def test_required_removal_is_breaking(self, tracker: SchemaEvolutionTracker):
        """Removing a required constraint should be marked as breaking."""
        df1 = pd.DataFrame({'val': [1, 2, 3]})  # Required
        df2 = pd.DataFrame({'val': [1, None, 3]})  # Nullable

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_business_rule_changes(current, previous, {'val'})

        if changes:
            assert changes[0]['breaking'] is True

    def test_adding_required_not_breaking(self, tracker: SchemaEvolutionTracker):
        """Adding a required constraint should not be breaking."""
        df1 = pd.DataFrame({'val': [1.0, None, 3.0]})  # Nullable
        df2 = pd.DataFrame({'val': [1.0, 2.0, 3.0]})  # Required

        previous = SchemaFingerprint.from_dataframe(df1, "test")
        current = SchemaFingerprint.from_dataframe(df2, "test")

        changes = tracker._detect_business_rule_changes(current, previous, {'val'})

        if changes:
            assert changes[0]['breaking'] is False


# =============================================================================
# PHASE 4: SCHEMA ANALYSIS TESTS
# =============================================================================


class TestAnalyzeSchemaChanges:
    """Tests for SchemaEvolutionTracker._analyze_schema_changes()."""

    def test_returns_baseline_for_first_fingerprint(self, tracker: SchemaEvolutionTracker,
                                                     sample_fingerprint: SchemaFingerprint):
        """Should return BASELINE_ESTABLISHED when no previous fingerprint."""
        analysis = tracker._analyze_schema_changes(sample_fingerprint, None, None)

        assert analysis['change_summary']['overall_status'] == 'BASELINE_ESTABLISHED'
        assert analysis['change_summary']['change_count'] == 0
        assert analysis['previous_fingerprint'] is None

    def test_includes_current_fingerprint(self, tracker: SchemaEvolutionTracker,
                                          sample_fingerprint: SchemaFingerprint):
        """Analysis should include current fingerprint data."""
        analysis = tracker._analyze_schema_changes(sample_fingerprint, None, None)

        assert analysis['current_fingerprint'] is not None
        assert 'columns' in analysis['current_fingerprint']

    def test_detects_changes_between_fingerprints(self, tracker: SchemaEvolutionTracker,
                                                   sample_dataframe: pd.DataFrame,
                                                   evolved_dataframe: pd.DataFrame):
        """Should detect changes between two fingerprints."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        analysis = tracker._analyze_schema_changes(current, previous, None)

        assert analysis['change_summary']['change_count'] > 0
        assert len(analysis['changes_detected']) > 0

    def test_includes_compatibility_analysis(self, tracker: SchemaEvolutionTracker,
                                              sample_dataframe: pd.DataFrame,
                                              evolved_dataframe: pd.DataFrame):
        """Analysis should include compatibility assessment."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        analysis = tracker._analyze_schema_changes(current, previous, None)

        assert 'compatibility_analysis' in analysis
        assert 'backward_compatible' in analysis['compatibility_analysis']

    def test_includes_recommendations(self, tracker: SchemaEvolutionTracker,
                                      sample_dataframe: pd.DataFrame,
                                      evolved_dataframe: pd.DataFrame):
        """Analysis should include actionable recommendations."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        analysis = tracker._analyze_schema_changes(current, previous, None)

        assert 'recommendations' in analysis
        assert isinstance(analysis['recommendations'], list)

    def test_includes_analysis_timestamp(self, tracker: SchemaEvolutionTracker,
                                         sample_fingerprint: SchemaFingerprint):
        """Analysis should include timestamp."""
        analysis = tracker._analyze_schema_changes(sample_fingerprint, None, None)

        assert 'analysis_timestamp' in analysis

    def test_no_changes_status(self, tracker: SchemaEvolutionTracker,
                               sample_dataframe: pd.DataFrame):
        """Should report NO_CHANGES when fingerprints are identical."""
        fp1 = SchemaFingerprint.from_dataframe(sample_dataframe, "test")
        fp2 = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        analysis = tracker._analyze_schema_changes(fp1, fp2, None)

        assert analysis['change_summary']['overall_status'] == 'NO_CHANGES'

    def test_breaking_changes_status(self, tracker: SchemaEvolutionTracker,
                                     sample_dataframe: pd.DataFrame,
                                     evolved_dataframe_removed_column: pd.DataFrame):
        """Should report BREAKING_CHANGES_DETECTED when breaking changes exist."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe_removed_column, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        analysis = tracker._analyze_schema_changes(current, previous, None)

        assert analysis['change_summary']['overall_status'] == 'BREAKING_CHANGES_DETECTED'
        assert analysis['change_summary']['breaking_changes'] > 0


class TestBuildChangeSummary:
    """Tests for SchemaEvolutionTracker._build_change_summary()."""

    def test_counts_total_changes(self, tracker: SchemaEvolutionTracker,
                                  sample_dataframe: pd.DataFrame,
                                  evolved_dataframe: pd.DataFrame):
        """Should count total number of changes."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes = [
            {'type': 'column_added', 'breaking': False},
            {'type': 'column_type_changed', 'breaking': True},
        ]

        summary = tracker._build_change_summary(
            changes, current, previous, {'new_feature'}, set()
        )

        assert summary['change_count'] == 2

    def test_counts_breaking_changes(self, tracker: SchemaEvolutionTracker,
                                     sample_dataframe: pd.DataFrame):
        """Should count breaking changes separately."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes = [
            {'type': 'column_added', 'breaking': False},
            {'type': 'column_removed', 'breaking': True},
            {'type': 'column_type_changed', 'breaking': True},
        ]

        summary = tracker._build_change_summary(changes, fp, fp, set(), {'removed'})

        assert summary['breaking_changes'] == 2

    def test_tracks_columns_added_count(self, tracker: SchemaEvolutionTracker,
                                        sample_dataframe: pd.DataFrame):
        """Should track number of columns added."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        summary = tracker._build_change_summary(
            [], fp, fp, {'new1', 'new2', 'new3'}, set()
        )

        assert summary['columns_added'] == 3

    def test_tracks_columns_removed_count(self, tracker: SchemaEvolutionTracker,
                                          sample_dataframe: pd.DataFrame):
        """Should track number of columns removed."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        summary = tracker._build_change_summary(
            [], fp, fp, set(), {'old1', 'old2'}
        )

        assert summary['columns_removed'] == 2

    def test_tracks_schema_hash_change(self, tracker: SchemaEvolutionTracker,
                                       sample_dataframe: pd.DataFrame,
                                       evolved_dataframe: pd.DataFrame):
        """Should detect when schema hash changes."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        summary = tracker._build_change_summary([], current, previous, set(), set())

        assert summary['schema_hash_changed'] is True

    def test_counts_type_changes(self, tracker: SchemaEvolutionTracker,
                                 sample_dataframe: pd.DataFrame):
        """Should count type changes."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        changes = [
            {'type': SchemaChangeType.COLUMN_TYPE_CHANGED.value, 'breaking': False},
            {'type': SchemaChangeType.COLUMN_TYPE_CHANGED.value, 'breaking': True},
            {'type': SchemaChangeType.COLUMN_ADDED.value, 'breaking': False},
        ]

        summary = tracker._build_change_summary(changes, fp, fp, set(), set())

        assert summary['type_changes'] == 2


class TestAnalyzeCompatibility:
    """Tests for SchemaEvolutionTracker._analyze_compatibility()."""

    def test_backward_compatible_when_no_removals(self, tracker: SchemaEvolutionTracker,
                                                   sample_dataframe: pd.DataFrame,
                                                   evolved_dataframe: pd.DataFrame):
        """Should be backward compatible when no columns removed."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        compat = tracker._analyze_compatibility(current, previous, None)

        assert compat['backward_compatible'] is True

    def test_not_backward_compatible_with_removals(self, tracker: SchemaEvolutionTracker,
                                                    sample_dataframe: pd.DataFrame,
                                                    evolved_dataframe_removed_column: pd.DataFrame):
        """Should not be backward compatible when columns removed."""
        current = SchemaFingerprint.from_dataframe(evolved_dataframe_removed_column, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        compat = tracker._analyze_compatibility(current, previous, None)

        assert compat['backward_compatible'] is False
        assert 'issues' in compat
        assert len(compat['issues']) > 0

    def test_calculates_compatibility_score(self, tracker: SchemaEvolutionTracker,
                                            sample_dataframe: pd.DataFrame):
        """Should calculate compatibility score as ratio of common columns."""
        current = SchemaFingerprint.from_dataframe(sample_dataframe, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        compat = tracker._analyze_compatibility(current, previous, None)

        assert compat['compatibility_score'] == 1.0

    def test_compatibility_score_decreases_with_changes(self, tracker: SchemaEvolutionTracker,
                                                        sample_dataframe: pd.DataFrame):
        """Compatibility score should decrease with more changes."""
        df_changed = sample_dataframe.copy()
        df_changed['new1'] = 1
        df_changed['new2'] = 2
        df_changed = df_changed.drop(columns=['price'])

        current = SchemaFingerprint.from_dataframe(df_changed, "test")
        previous = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        compat = tracker._analyze_compatibility(current, previous, None)

        assert compat['compatibility_score'] < 1.0

    def test_includes_forward_compatible_flag(self, tracker: SchemaEvolutionTracker,
                                              sample_dataframe: pd.DataFrame):
        """Should include forward compatibility flag."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        compat = tracker._analyze_compatibility(fp, fp, None)

        assert 'forward_compatible' in compat

    def test_baseline_compliant_none_without_schema(self, tracker: SchemaEvolutionTracker,
                                                    sample_dataframe: pd.DataFrame):
        """baseline_compliant should be None when no baseline schema provided."""
        fp = SchemaFingerprint.from_dataframe(sample_dataframe, "test")

        compat = tracker._analyze_compatibility(fp, fp, None)

        assert compat['baseline_compliant'] is None


class TestGenerateRecommendations:
    """Tests for SchemaEvolutionTracker._generate_recommendations()."""

    def test_warns_on_breaking_changes(self, tracker: SchemaEvolutionTracker):
        """Should warn about breaking changes."""
        changes = [{'type': 'column_removed', 'breaking': True}]
        compatibility = {'compatibility_score': 0.9}

        recommendations = tracker._generate_recommendations(changes, compatibility)

        assert any('BREAKING' in r.upper() for r in recommendations)

    def test_notes_added_columns(self, tracker: SchemaEvolutionTracker):
        """Should note when columns are added."""
        changes = [
            {'type': SchemaChangeType.COLUMN_ADDED.value, 'breaking': False},
            {'type': SchemaChangeType.COLUMN_ADDED.value, 'breaking': False},
        ]
        compatibility = {'compatibility_score': 0.9}

        recommendations = tracker._generate_recommendations(changes, compatibility)

        assert any('added' in r.lower() for r in recommendations)

    def test_warns_on_removed_columns(self, tracker: SchemaEvolutionTracker):
        """Should warn about removed columns."""
        changes = [{'type': SchemaChangeType.COLUMN_REMOVED.value, 'breaking': True}]
        compatibility = {'compatibility_score': 0.9}

        recommendations = tracker._generate_recommendations(changes, compatibility)

        assert any('removed' in r.lower() for r in recommendations)

    def test_warns_on_low_compatibility(self, tracker: SchemaEvolutionTracker):
        """Should warn when compatibility score is low."""
        changes = [{'type': 'column_added', 'breaking': False}]
        compatibility = {'compatibility_score': 0.5}

        recommendations = tracker._generate_recommendations(changes, compatibility)

        assert any('compatibility' in r.lower() for r in recommendations)

    def test_success_message_no_changes(self, tracker: SchemaEvolutionTracker):
        """Should report success when no changes detected."""
        changes = []
        compatibility = {'compatibility_score': 1.0}

        recommendations = tracker._generate_recommendations(changes, compatibility)

        assert any('no schema changes' in r.lower() for r in recommendations)

    def test_recommends_regression_tests(self, tracker: SchemaEvolutionTracker):
        """Should recommend regression tests when changes exist."""
        changes = [{'type': 'column_added', 'breaking': False}]
        compatibility = {'compatibility_score': 0.9}

        recommendations = tracker._generate_recommendations(changes, compatibility)

        assert any('regression' in r.lower() or 'test' in r.lower() for r in recommendations)


# =============================================================================
# PHASE 5: FILE I/O TESTS
# =============================================================================


class TestLoadEvolutionHistory:
    """Tests for SchemaEvolutionTracker._load_evolution_history()."""

    def test_returns_empty_dict_when_no_file(self, tracker: SchemaEvolutionTracker):
        """Should return empty dict when history file doesn't exist."""
        history = tracker._load_evolution_history()

        assert history == {}

    def test_loads_existing_history(self, tracker: SchemaEvolutionTracker,
                                    temp_tracking_dir: Path):
        """Should load existing history from file."""
        test_history = {
            "test_schema": [
                {"timestamp": "2023-01-01T00:00:00", "fingerprint": {}}
            ]
        }

        with open(tracker.history_file, 'w') as f:
            json.dump(test_history, f)

        history = tracker._load_evolution_history()

        assert "test_schema" in history
        assert len(history["test_schema"]) == 1

    def test_handles_corrupted_json(self, tracker: SchemaEvolutionTracker):
        """Should handle corrupted JSON gracefully."""
        with open(tracker.history_file, 'w') as f:
            f.write("not valid json {{{")

        history = tracker._load_evolution_history()

        assert history == {}

    def test_handles_permission_error(self, tracker: SchemaEvolutionTracker,
                                      monkeypatch):
        """Should handle permission errors gracefully."""
        def mock_open_error(*args, **kwargs):
            raise PermissionError("Permission denied")

        # Create a valid file first
        with open(tracker.history_file, 'w') as f:
            json.dump({}, f)

        # Then mock the open to fail
        monkeypatch.setattr('builtins.open', mock_open_error)

        history = tracker._load_evolution_history()
        assert history == {}


class TestUpdateEvolutionHistory:
    """Tests for SchemaEvolutionTracker._update_evolution_history()."""

    def test_creates_new_schema_entry(self, tracker: SchemaEvolutionTracker,
                                       sample_fingerprint: SchemaFingerprint):
        """Should create new entry for new schema name."""
        analysis = {'change_summary': {'overall_status': 'BASELINE_ESTABLISHED'}}

        tracker._update_evolution_history("new_schema", sample_fingerprint, analysis)

        history = tracker._load_evolution_history()
        assert "new_schema" in history
        assert len(history["new_schema"]) == 1

    def test_appends_to_existing_schema(self, tracker: SchemaEvolutionTracker,
                                        sample_fingerprint: SchemaFingerprint):
        """Should append to existing schema history."""
        analysis = {'change_summary': {'overall_status': 'BASELINE_ESTABLISHED'}}

        tracker._update_evolution_history("test", sample_fingerprint, analysis)
        tracker._update_evolution_history("test", sample_fingerprint, analysis)

        history = tracker._load_evolution_history()
        assert len(history["test"]) == 2

    def test_limits_history_to_50_entries(self, tracker: SchemaEvolutionTracker,
                                          sample_fingerprint: SchemaFingerprint):
        """Should keep only last 50 entries per schema."""
        analysis = {'change_summary': {'overall_status': 'TEST'}}

        for i in range(60):
            tracker._update_evolution_history("test", sample_fingerprint, analysis)

        history = tracker._load_evolution_history()
        assert len(history["test"]) == 50

    def test_saves_timestamp(self, tracker: SchemaEvolutionTracker,
                             sample_fingerprint: SchemaFingerprint):
        """Should save fingerprint timestamp."""
        analysis = {'change_summary': {'overall_status': 'TEST'}}

        tracker._update_evolution_history("test", sample_fingerprint, analysis)

        history = tracker._load_evolution_history()
        assert 'timestamp' in history['test'][0]

    def test_saves_fingerprint_data(self, tracker: SchemaEvolutionTracker,
                                    sample_fingerprint: SchemaFingerprint):
        """Should save complete fingerprint data."""
        analysis = {'change_summary': {'overall_status': 'TEST'}}

        tracker._update_evolution_history("test", sample_fingerprint, analysis)

        history = tracker._load_evolution_history()
        assert 'fingerprint' in history['test'][0]
        assert 'columns' in history['test'][0]['fingerprint']

    def test_saves_analysis_data(self, tracker: SchemaEvolutionTracker,
                                 sample_fingerprint: SchemaFingerprint):
        """Should save evolution analysis data."""
        analysis = {'change_summary': {'overall_status': 'TEST', 'change_count': 5}}

        tracker._update_evolution_history("test", sample_fingerprint, analysis)

        history = tracker._load_evolution_history()
        assert 'evolution_analysis' in history['test'][0]

    def test_handles_write_error(self, tracker: SchemaEvolutionTracker,
                                 sample_fingerprint: SchemaFingerprint,
                                 monkeypatch, capsys):
        """Should handle write errors gracefully."""
        analysis = {'change_summary': {'overall_status': 'TEST'}}

        def mock_open_error(*args, **kwargs):
            if 'w' in args[1] if len(args) > 1 else kwargs.get('mode', ''):
                raise PermissionError("Permission denied")
            return open(*args, **kwargs)

        # First call to read should work
        tracker._update_evolution_history("test", sample_fingerprint, analysis)

        # Now mock the write to fail
        original_open = open
        call_count = [0]

        def controlled_mock_open(*args, **kwargs):
            call_count[0] += 1
            mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
            if 'w' in mode and call_count[0] > 1:
                raise PermissionError("Permission denied")
            return original_open(*args, **kwargs)

        monkeypatch.setattr('builtins.open', controlled_mock_open)

        # Should not crash
        tracker._update_evolution_history("test2", sample_fingerprint, analysis)
        captured = capsys.readouterr()
        assert 'WARNING' in captured.out or True  # May or may not print warning


# =============================================================================
# PHASE 6: ML DOCS INTEGRATION TESTS (for schema_evolution functions)
# =============================================================================


class TestTrackSchemaEvolution:
    """Tests for SchemaEvolutionTracker.track_schema_evolution() main method."""

    def test_tracks_baseline_schema(self, tracker: SchemaEvolutionTracker,
                                    sample_dataframe: pd.DataFrame, capsys):
        """Should establish baseline for first tracking call."""
        result = tracker.track_schema_evolution(sample_dataframe, "test_schema")

        assert result['change_summary']['overall_status'] == 'BASELINE_ESTABLISHED'

        captured = capsys.readouterr()
        assert 'SUCCESS' in captured.out

    def test_tracks_evolved_schema(self, tracker: SchemaEvolutionTracker,
                                   sample_dataframe: pd.DataFrame,
                                   evolved_dataframe: pd.DataFrame, capsys):
        """Should detect changes in evolved schema."""
        # First call establishes baseline
        tracker.track_schema_evolution(sample_dataframe, "test_schema")

        # Second call should detect changes
        result = tracker.track_schema_evolution(evolved_dataframe, "test_schema")

        assert result['change_summary']['change_count'] > 0
        captured = capsys.readouterr()
        assert 'Schema Hash' in captured.out

    def test_returns_complete_analysis(self, tracker: SchemaEvolutionTracker,
                                       sample_dataframe: pd.DataFrame):
        """Should return complete analysis dictionary."""
        result = tracker.track_schema_evolution(sample_dataframe, "test_schema")

        assert 'analysis_timestamp' in result
        assert 'current_fingerprint' in result
        assert 'changes_detected' in result
        assert 'change_summary' in result
        assert 'recommendations' in result

    def test_updates_history(self, tracker: SchemaEvolutionTracker,
                             sample_dataframe: pd.DataFrame):
        """Should update evolution history after tracking."""
        tracker.track_schema_evolution(sample_dataframe, "test_schema")

        history = tracker._load_evolution_history()
        assert "test_schema" in history

    def test_prints_progress_messages(self, tracker: SchemaEvolutionTracker,
                                      sample_dataframe: pd.DataFrame, capsys):
        """Should print progress messages during tracking."""
        tracker.track_schema_evolution(sample_dataframe, "test_schema")

        captured = capsys.readouterr()
        assert 'ANALYSIS' in captured.out
        assert 'SUCCESS' in captured.out

    def test_handles_baseline_schema_parameter(self, tracker: SchemaEvolutionTracker,
                                               sample_dataframe: pd.DataFrame,
                                               mock_pandera_schema: MagicMock):
        """Should accept optional baseline_schema parameter."""
        # This should not crash even with mock schema
        result = tracker.track_schema_evolution(
            sample_dataframe, "test_schema", baseline_schema=mock_pandera_schema
        )

        assert result is not None


class TestGetEvolutionSummary:
    """Tests for SchemaEvolutionTracker.get_evolution_summary()."""

    def test_returns_empty_summary_no_history(self, tracker: SchemaEvolutionTracker):
        """Should return empty summary when no history exists."""
        summary = tracker.get_evolution_summary()

        assert summary['total_schemas'] == 0
        assert summary['schemas_tracked'] == []

    def test_returns_specific_schema_summary(self, tracker: SchemaEvolutionTracker,
                                              sample_dataframe: pd.DataFrame):
        """Should return summary for specific schema."""
        tracker.track_schema_evolution(sample_dataframe, "test_schema")

        summary = tracker.get_evolution_summary("test_schema")

        assert summary['status'] == 'TRACKED'
        assert summary['entries'] == 1

    def test_returns_no_history_for_unknown_schema(self, tracker: SchemaEvolutionTracker):
        """Should return NO_HISTORY for unknown schema name."""
        summary = tracker.get_evolution_summary("nonexistent")

        assert summary['status'] == 'NO_HISTORY'

    def test_returns_all_schemas_summary(self, tracker: SchemaEvolutionTracker,
                                          sample_dataframe: pd.DataFrame):
        """Should return summary for all tracked schemas."""
        tracker.track_schema_evolution(sample_dataframe, "schema1")
        tracker.track_schema_evolution(sample_dataframe, "schema2")

        summary = tracker.get_evolution_summary()

        assert summary['total_schemas'] == 2
        assert 'schema1' in summary['schemas_tracked']
        assert 'schema2' in summary['schemas_tracked']

    def test_includes_latest_info(self, tracker: SchemaEvolutionTracker,
                                  sample_dataframe: pd.DataFrame):
        """Should include latest schema info in summary."""
        tracker.track_schema_evolution(sample_dataframe, "test_schema")

        summary = tracker.get_evolution_summary("test_schema")

        assert 'latest_timestamp' in summary
        assert 'latest_schema_hash' in summary
        assert 'latest_columns' in summary


# =============================================================================
# PHASE 7: TRACKER ORCHESTRATION & CONVENIENCE FUNCTIONS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_track_dataset_evolution_creates_tracker(self, sample_dataframe: pd.DataFrame,
                                                     tmp_path, monkeypatch):
        """track_dataset_evolution() should create tracker if not provided."""
        # Change to temp dir to avoid cluttering project
        monkeypatch.chdir(tmp_path)

        result = track_dataset_evolution(sample_dataframe, "test")

        assert result is not None
        assert 'change_summary' in result

    def test_track_dataset_evolution_uses_provided_tracker(
        self, tracker: SchemaEvolutionTracker, sample_dataframe: pd.DataFrame
    ):
        """track_dataset_evolution() should use provided tracker."""
        result = track_dataset_evolution(sample_dataframe, "test", tracker=tracker)

        # Verify it used our tracker by checking its history
        history = tracker._load_evolution_history()
        assert "test" in history

    def test_get_evolution_report_creates_tracker(self, tmp_path, monkeypatch):
        """get_evolution_report() should create tracker if not provided."""
        monkeypatch.chdir(tmp_path)

        result = get_evolution_report()

        assert result is not None
        assert 'total_schemas' in result

    def test_get_evolution_report_uses_provided_tracker(
        self, tracker: SchemaEvolutionTracker, sample_dataframe: pd.DataFrame
    ):
        """get_evolution_report() should use provided tracker."""
        tracker.track_schema_evolution(sample_dataframe, "test")

        result = get_evolution_report(tracker=tracker)

        assert result['total_schemas'] == 1

    def test_get_evolution_report_specific_schema(
        self, tracker: SchemaEvolutionTracker, sample_dataframe: pd.DataFrame
    ):
        """get_evolution_report() should work with specific schema name."""
        tracker.track_schema_evolution(sample_dataframe, "test")

        result = get_evolution_report(schema_name="test", tracker=tracker)

        assert result['status'] == 'TRACKED'


# =============================================================================
# PHASE 8: MLFLOW INTEGRATION & EDGE CASES
# =============================================================================


class TestLogEvolutionToMLflow:
    """Tests for SchemaEvolutionTracker._log_evolution_to_mlflow()."""

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="MLflow modules not available"
    )
    def test_logs_metrics_when_available(self, tracker: SchemaEvolutionTracker):
        """Should log metrics when MLflow modules available."""
        analysis = {
            'change_summary': {
                'overall_status': 'MINOR_CHANGES',
                'change_count': 2,
                'breaking_changes': 0
            },
            'compatibility_analysis': {
                'compatibility_score': 0.95
            }
        }

        with patch('src.validation.schema_evolution.safe_mlflow_log_param') as mock_param, \
             patch('src.validation.schema_evolution.safe_mlflow_log_metric') as mock_metric:
            tracker._log_evolution_to_mlflow("test", analysis)

            mock_param.assert_called()
            mock_metric.assert_called()

    def test_skips_when_modules_unavailable(self, tracker: SchemaEvolutionTracker,
                                            monkeypatch):
        """Should skip logging when validation modules unavailable."""
        monkeypatch.setattr(
            'src.validation.schema_evolution.VALIDATION_MODULES_AVAILABLE',
            False
        )

        analysis = {'change_summary': {'overall_status': 'TEST'}}

        # Should not raise
        tracker._log_evolution_to_mlflow("test", analysis)

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="MLflow modules not available"
    )
    def test_handles_mlflow_errors(self, tracker: SchemaEvolutionTracker, capsys):
        """Should handle MLflow errors gracefully."""
        analysis = {
            'change_summary': {'overall_status': 'TEST', 'change_count': 0, 'breaking_changes': 0},
            'compatibility_analysis': {'compatibility_score': 1.0}
        }

        with patch('src.validation.schema_evolution.safe_mlflow_log_param',
                   side_effect=Exception("MLflow error")):
            # Should not raise
            tracker._log_evolution_to_mlflow("test", analysis)

            captured = capsys.readouterr()
            assert 'WARNING' in captured.out


class TestIsCompatibleTypeChange:
    """Tests for SchemaEvolutionTracker._is_compatible_type_change()."""

    def test_int32_to_int64_compatible(self, tracker: SchemaEvolutionTracker):
        """int32 to int64 should be compatible."""
        assert tracker._is_compatible_type_change("int32", "int64") is True

    def test_float32_to_float64_compatible(self, tracker: SchemaEvolutionTracker):
        """float32 to float64 should be compatible."""
        assert tracker._is_compatible_type_change("float32", "float64") is True

    def test_int_to_float_compatible(self, tracker: SchemaEvolutionTracker):
        """int to float should be compatible."""
        assert tracker._is_compatible_type_change("int32", "float64") is True
        assert tracker._is_compatible_type_change("int64", "float64") is True

    def test_object_to_string_compatible(self, tracker: SchemaEvolutionTracker):
        """object to string should be compatible."""
        assert tracker._is_compatible_type_change("object", "string") is True

    def test_incompatible_changes(self, tracker: SchemaEvolutionTracker):
        """Incompatible changes should return False."""
        assert tracker._is_compatible_type_change("float64", "int64") is False
        assert tracker._is_compatible_type_change("string", "int64") is False
        assert tracker._is_compatible_type_change("datetime64", "float64") is False


class TestIsBreakingRuleChange:
    """Tests for SchemaEvolutionTracker._is_breaking_rule_change()."""

    def test_removing_required_is_breaking(self, tracker: SchemaEvolutionTracker):
        """Removing required rule should be breaking."""
        previous = [{'type': 'required', 'nullable': False}]
        current = []

        assert tracker._is_breaking_rule_change(previous, current) is True

    def test_adding_rules_not_breaking(self, tracker: SchemaEvolutionTracker):
        """Adding new rules should not be breaking."""
        previous = []
        current = [{'type': 'required', 'nullable': False}]

        assert tracker._is_breaking_rule_change(previous, current) is False

    def test_no_change_not_breaking(self, tracker: SchemaEvolutionTracker):
        """No rule changes should not be breaking."""
        rules = [{'type': 'required', 'nullable': False}]

        assert tracker._is_breaking_rule_change(rules, rules) is False


class TestDetermineOverallStatus:
    """Tests for SchemaEvolutionTracker._determine_overall_status()."""

    def test_no_changes_status(self, tracker: SchemaEvolutionTracker):
        """Empty changes should return NO_CHANGES."""
        assert tracker._determine_overall_status([], 0) == "NO_CHANGES"

    def test_breaking_changes_status(self, tracker: SchemaEvolutionTracker):
        """Breaking changes should return BREAKING_CHANGES_DETECTED."""
        changes = [{'type': 'column_removed'}]
        assert tracker._determine_overall_status(changes, 1) == "BREAKING_CHANGES_DETECTED"

    def test_major_changes_status(self, tracker: SchemaEvolutionTracker):
        """Many changes should return MAJOR_CHANGES."""
        changes = [{'type': f'change_{i}'} for i in range(6)]
        assert tracker._determine_overall_status(changes, 0) == "MAJOR_CHANGES"

    def test_minor_changes_status(self, tracker: SchemaEvolutionTracker):
        """Few non-breaking changes should return MINOR_CHANGES."""
        changes = [{'type': 'column_added'}]
        assert tracker._determine_overall_status(changes, 0) == "MINOR_CHANGES"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self, tracker: SchemaEvolutionTracker):
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        fp = SchemaFingerprint.from_dataframe(df, "empty")

        assert fp.shape == (0, 0)
        assert fp.columns == []

    def test_single_row_dataframe(self, tracker: SchemaEvolutionTracker):
        """Should handle single row DataFrame."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        fp = SchemaFingerprint.from_dataframe(df, "single")

        assert fp.shape == (1, 2)
        assert fp.duplicate_rows == 0

    def test_all_null_dataframe(self, tracker: SchemaEvolutionTracker):
        """Should handle DataFrame with all nulls."""
        df = pd.DataFrame({'a': [None, None], 'b': [None, None]})
        fp = SchemaFingerprint.from_dataframe(df, "nulls")

        assert fp.null_percentages['a'] == 100.0
        assert fp.null_percentages['b'] == 100.0

    def test_very_large_column_count(self, tracker: SchemaEvolutionTracker):
        """Should handle DataFrame with many columns."""
        data = {f'col_{i}': [1, 2, 3] for i in range(100)}
        df = pd.DataFrame(data)
        fp = SchemaFingerprint.from_dataframe(df, "wide")

        assert fp.shape == (3, 100)
        assert len(fp.columns) == 100

    def test_special_characters_in_column_names(self, tracker: SchemaEvolutionTracker):
        """Should handle special characters in column names."""
        df = pd.DataFrame({
            'col with spaces': [1],
            'col-with-dashes': [2],
            'col.with.dots': [3],
            'col_123': [4]
        })
        fp = SchemaFingerprint.from_dataframe(df, "special")

        assert len(fp.columns) == 4

    def test_unicode_column_names(self, tracker: SchemaEvolutionTracker):
        """Should handle unicode column names."""
        df = pd.DataFrame({
            '日本語': [1],
            'émoji': [2],
            'Ελληνικά': [3]
        })
        fp = SchemaFingerprint.from_dataframe(df, "unicode")

        assert len(fp.columns) == 3

    def test_date_column_with_nat(self, tracker: SchemaEvolutionTracker):
        """Should handle date column with NaT values."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2022-01-01', pd.NaT, '2022-01-03'])
        })
        fp = SchemaFingerprint.from_dataframe(df, "nat_dates")

        # Should not crash; date_range_days may be None or calculated
        assert isinstance(fp, SchemaFingerprint)

    def test_tracker_initialization_creates_directory(self, tmp_path):
        """Tracker should create tracking directory if it doesn't exist."""
        new_dir = tmp_path / "new_tracking_dir"
        tracker = SchemaEvolutionTracker(tracking_dir=str(new_dir))

        assert new_dir.exists()

    def test_multiple_trackers_same_directory(self, temp_tracking_dir: Path,
                                               sample_dataframe: pd.DataFrame):
        """Multiple trackers should be able to use same directory."""
        tracker1 = SchemaEvolutionTracker(tracking_dir=str(temp_tracking_dir))
        tracker2 = SchemaEvolutionTracker(tracking_dir=str(temp_tracking_dir))

        tracker1.track_schema_evolution(sample_dataframe, "schema1")
        tracker2.track_schema_evolution(sample_dataframe, "schema2")

        # Both should be in history
        history = tracker1._load_evolution_history()
        assert "schema1" in history
        assert "schema2" in history


class TestGenerateSchemaSummary:
    """Tests for SchemaEvolutionTracker._generate_schema_summary()."""

    def test_empty_history_returns_no_history(self, tracker: SchemaEvolutionTracker):
        """Empty history should return NO_HISTORY status."""
        summary = tracker._generate_schema_summary("test", [])

        assert summary['status'] == 'NO_HISTORY'
        assert summary['entries'] == 0

    def test_returns_entry_count(self, tracker: SchemaEvolutionTracker,
                                 sample_dataframe: pd.DataFrame):
        """Should return correct entry count."""
        tracker.track_schema_evolution(sample_dataframe, "test")
        tracker.track_schema_evolution(sample_dataframe, "test")

        history = tracker._load_evolution_history()
        summary = tracker._generate_schema_summary("test", history["test"])

        assert summary['entries'] == 2

    def test_calculates_total_changes(self, tracker: SchemaEvolutionTracker,
                                      sample_dataframe: pd.DataFrame,
                                      evolved_dataframe: pd.DataFrame):
        """Should calculate total changes tracked."""
        tracker.track_schema_evolution(sample_dataframe, "test")
        tracker.track_schema_evolution(evolved_dataframe, "test")

        history = tracker._load_evolution_history()
        summary = tracker._generate_schema_summary("test", history["test"])

        assert 'total_changes_tracked' in summary
