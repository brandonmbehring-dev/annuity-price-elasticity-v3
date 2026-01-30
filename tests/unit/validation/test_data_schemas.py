"""
Tests for src.validation.data_schemas module.

Comprehensive tests for Pandera DataFrame schema validation including:
- Schema instantiation and basic validation
- Valid data passing validation
- Missing columns failing validation
- Wrong types failing validation
- Edge cases: nulls, duplicates, empty datasets
- DataFrameValidator methods
- SchemaAwareDVCTracker methods
- MLflow integration

Coverage Target: 60%+ of data_schemas.py (828 lines)
"""

import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from src.validation.data_schemas import (
    FINAL_DATASET_SCHEMA,
    FORECAST_RESULTS_SCHEMA,
    FEATURE_SELECTION_RESULTS_SCHEMA,
    DataFrameValidator,
    SchemaAwareDVCTracker,
    load_validated_dataset,
    validate_and_save_dataset,
    load_and_validate_with_dvc_context,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def valid_final_dataset() -> pd.DataFrame:
    """Create valid final dataset matching FINAL_DATASET_SCHEMA."""
    np.random.seed(42)
    n_obs = 100

    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n_obs, freq='D'),
        'sales': np.random.uniform(50000, 200000, n_obs),
        'sales_target_t0': np.random.uniform(50000, 200000, n_obs),
        'prudential_rate_t0': np.random.uniform(1, 5, n_obs),
        'competitor_weighted_t2': np.random.uniform(-2, 2, n_obs),
        'competitor_top5_t3': np.random.uniform(-4, 2, n_obs),
        'sales_target_contract_t5': np.random.uniform(40000, 180000, n_obs),
        'weight': np.random.uniform(0.1, 1.0, n_obs)
    })


@pytest.fixture
def valid_forecast_results() -> pd.DataFrame:
    """Create valid forecast results matching FORECAST_RESULTS_SCHEMA."""
    np.random.seed(42)
    n_obs = 50

    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_obs, freq='D'),
        'y_true': np.random.uniform(50000, 200000, n_obs),
        'y_predict': np.random.uniform(45000, 210000, n_obs),
        'abs_pct_error': np.random.uniform(0.05, 0.25, n_obs)
    })


@pytest.fixture
def valid_feature_selection_results() -> pd.DataFrame:
    """Create valid feature selection results matching FEATURE_SELECTION_RESULTS_SCHEMA."""
    return pd.DataFrame({
        'feature_name': ['prudential_rate_t0', 'competitor_weighted_t2', 'spread', 'vix'],
        'coefficient': [0.15, -0.08, 0.02, -0.01],
        'p_value': [0.01, 0.03, 0.15, 0.40],
        'selected': [True, True, False, False]
    })


@pytest.fixture
def temp_dir():
    """Create temporary directory for file operations."""
    temp_path = tempfile.mkdtemp(prefix="test_data_schemas_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


# =============================================================================
# SCHEMA INSTANTIATION TESTS
# =============================================================================


class TestSchemaInstantiation:
    """Tests for schema instantiation and basic structure."""

    def test_final_dataset_schema_exists(self):
        """FINAL_DATASET_SCHEMA should be a valid DataFrameSchema."""
        assert isinstance(FINAL_DATASET_SCHEMA, pa.DataFrameSchema)

    def test_forecast_results_schema_exists(self):
        """FORECAST_RESULTS_SCHEMA should be a valid DataFrameSchema."""
        assert isinstance(FORECAST_RESULTS_SCHEMA, pa.DataFrameSchema)

    def test_feature_selection_schema_exists(self):
        """FEATURE_SELECTION_RESULTS_SCHEMA should be a valid DataFrameSchema."""
        assert isinstance(FEATURE_SELECTION_RESULTS_SCHEMA, pa.DataFrameSchema)

    def test_final_dataset_schema_has_required_columns(self):
        """FINAL_DATASET_SCHEMA should define all required columns."""
        expected_columns = {
            'date', 'sales', 'sales_target_t0', 'prudential_rate_t0',
            'competitor_weighted_t2', 'competitor_top5_t3',
            'sales_target_contract_t5', 'weight'
        }
        schema_columns = set(FINAL_DATASET_SCHEMA.columns.keys())
        assert expected_columns == schema_columns

    def test_forecast_results_schema_has_required_columns(self):
        """FORECAST_RESULTS_SCHEMA should define all required columns."""
        expected_columns = {'date', 'y_true', 'y_predict', 'abs_pct_error'}
        schema_columns = set(FORECAST_RESULTS_SCHEMA.columns.keys())
        assert expected_columns == schema_columns

    def test_feature_selection_schema_has_required_columns(self):
        """FEATURE_SELECTION_RESULTS_SCHEMA should define all required columns."""
        expected_columns = {'feature_name', 'coefficient', 'p_value', 'selected'}
        schema_columns = set(FEATURE_SELECTION_RESULTS_SCHEMA.columns.keys())
        assert expected_columns == schema_columns


# =============================================================================
# VALID DATA PASSES VALIDATION TESTS
# =============================================================================


class TestValidDataPassesValidation:
    """Tests that valid data passes schema validation."""

    def test_valid_final_dataset_passes(self, valid_final_dataset):
        """Valid final dataset should pass FINAL_DATASET_SCHEMA validation."""
        validated = FINAL_DATASET_SCHEMA.validate(valid_final_dataset)
        assert len(validated) == len(valid_final_dataset)

    def test_valid_forecast_results_passes(self, valid_forecast_results):
        """Valid forecast results should pass FORECAST_RESULTS_SCHEMA validation."""
        validated = FORECAST_RESULTS_SCHEMA.validate(valid_forecast_results)
        assert len(validated) == len(valid_forecast_results)

    def test_valid_feature_selection_passes(self, valid_feature_selection_results):
        """Valid feature selection results should pass schema validation."""
        validated = FEATURE_SELECTION_RESULTS_SCHEMA.validate(valid_feature_selection_results)
        assert len(validated) == len(valid_feature_selection_results)

    def test_dataframe_validator_final_dataset(self, valid_final_dataset):
        """DataFrameValidator.validate_final_dataset should accept valid data."""
        validated = DataFrameValidator.validate_final_dataset(valid_final_dataset)
        assert len(validated) == len(valid_final_dataset)

    def test_dataframe_validator_forecast_results(self, valid_forecast_results):
        """DataFrameValidator.validate_forecast_results should accept valid data."""
        validated = DataFrameValidator.validate_forecast_results(valid_forecast_results)
        assert len(validated) == len(valid_forecast_results)

    def test_dataframe_validator_feature_selection(self, valid_feature_selection_results):
        """DataFrameValidator.validate_feature_selection_results should accept valid data."""
        validated = DataFrameValidator.validate_feature_selection_results(
            valid_feature_selection_results
        )
        assert len(validated) == len(valid_feature_selection_results)


# =============================================================================
# MISSING COLUMNS FAIL VALIDATION TESTS
# =============================================================================


class TestMissingColumnsFail:
    """Tests that missing required columns fail validation."""

    def test_missing_date_column_fails(self, valid_final_dataset):
        """Missing 'date' column should fail validation."""
        df_missing = valid_final_dataset.drop(columns=['date'])
        with pytest.raises(pa.errors.SchemaError):
            FINAL_DATASET_SCHEMA.validate(df_missing)

    def test_missing_sales_column_fails(self, valid_final_dataset):
        """Missing 'sales' column should fail validation."""
        df_missing = valid_final_dataset.drop(columns=['sales'])
        with pytest.raises(pa.errors.SchemaError):
            FINAL_DATASET_SCHEMA.validate(df_missing)

    def test_missing_prudential_rate_fails(self, valid_final_dataset):
        """Missing 'prudential_rate_t0' column should fail validation."""
        df_missing = valid_final_dataset.drop(columns=['prudential_rate_t0'])
        with pytest.raises(pa.errors.SchemaError):
            FINAL_DATASET_SCHEMA.validate(df_missing)

    def test_missing_y_true_fails_forecast(self, valid_forecast_results):
        """Missing 'y_true' column should fail forecast schema validation."""
        df_missing = valid_forecast_results.drop(columns=['y_true'])
        with pytest.raises(pa.errors.SchemaError):
            FORECAST_RESULTS_SCHEMA.validate(df_missing)

    def test_missing_feature_name_fails_selection(self, valid_feature_selection_results):
        """Missing 'feature_name' column should fail feature selection schema."""
        df_missing = valid_feature_selection_results.drop(columns=['feature_name'])
        with pytest.raises(pa.errors.SchemaError):
            FEATURE_SELECTION_RESULTS_SCHEMA.validate(df_missing)


# =============================================================================
# WRONG TYPES FAIL VALIDATION TESTS
# =============================================================================


class TestWrongTypesFail:
    """Tests that wrong column types fail validation."""

    def test_sales_negative_fails(self, valid_final_dataset):
        """Negative sales values should fail validation (Check.ge(0))."""
        df_invalid = valid_final_dataset.copy()
        df_invalid.loc[0, 'sales'] = -1000
        with pytest.raises(pa.errors.SchemaError):
            FINAL_DATASET_SCHEMA.validate(df_invalid)

    def test_prudential_rate_out_of_range_fails(self, valid_final_dataset):
        """Prudential rate > 20 should fail validation (Check.between(0, 20))."""
        df_invalid = valid_final_dataset.copy()
        df_invalid.loc[0, 'prudential_rate_t0'] = 25.0
        with pytest.raises(pa.errors.SchemaError):
            FINAL_DATASET_SCHEMA.validate(df_invalid)

    def test_weight_out_of_range_fails(self, valid_final_dataset):
        """Weight > 1 should fail validation (Check.between(0, 1))."""
        df_invalid = valid_final_dataset.copy()
        df_invalid.loc[0, 'weight'] = 1.5
        with pytest.raises(pa.errors.SchemaError):
            FINAL_DATASET_SCHEMA.validate(df_invalid)

    def test_y_true_negative_fails_forecast(self, valid_forecast_results):
        """Negative y_true values should fail forecast validation."""
        df_invalid = valid_forecast_results.copy()
        df_invalid.loc[0, 'y_true'] = -500
        with pytest.raises(pa.errors.SchemaError):
            FORECAST_RESULTS_SCHEMA.validate(df_invalid)

    def test_abs_pct_error_out_of_range_fails(self, valid_forecast_results):
        """abs_pct_error > 5 should fail forecast validation."""
        df_invalid = valid_forecast_results.copy()
        df_invalid.loc[0, 'abs_pct_error'] = 6.0
        with pytest.raises(pa.errors.SchemaError):
            FORECAST_RESULTS_SCHEMA.validate(df_invalid)

    def test_p_value_out_of_range_fails(self, valid_feature_selection_results):
        """p_value > 1 should fail feature selection validation."""
        df_invalid = valid_feature_selection_results.copy()
        df_invalid.loc[0, 'p_value'] = 1.5
        with pytest.raises(pa.errors.SchemaError):
            FEATURE_SELECTION_RESULTS_SCHEMA.validate(df_invalid)


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases: nulls, duplicates, empty datasets."""

    def test_nullable_columns_accept_null(self, valid_final_dataset):
        """Nullable columns (weight, sales_target_contract_t5) should accept null."""
        df_with_nulls = valid_final_dataset.copy()
        df_with_nulls.loc[0, 'weight'] = None
        df_with_nulls.loc[1, 'sales_target_contract_t5'] = None

        # Should pass (these columns are nullable=True)
        validated = FINAL_DATASET_SCHEMA.validate(df_with_nulls)
        assert pd.isna(validated.loc[0, 'weight'])
        assert pd.isna(validated.loc[1, 'sales_target_contract_t5'])

    def test_non_nullable_columns_reject_null(self, valid_final_dataset):
        """Non-nullable columns should reject null values."""
        df_with_nulls = valid_final_dataset.copy()
        df_with_nulls.loc[0, 'sales'] = None  # sales is nullable=False

        with pytest.raises(pa.errors.SchemaError):
            FINAL_DATASET_SCHEMA.validate(df_with_nulls)

    def test_dataset_too_small_fails_validator(self):
        """Dataset with < 50 rows should fail DataFrameValidator business logic."""
        np.random.seed(42)
        small_df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=30, freq='D'),
            'sales': np.random.uniform(50000, 200000, 30),
            'sales_target_t0': np.random.uniform(50000, 200000, 30),
            'prudential_rate_t0': np.random.uniform(1, 5, 30),
            'competitor_weighted_t2': np.random.uniform(-2, 2, 30),
            'competitor_top5_t3': np.random.uniform(-4, 2, 30),
            'sales_target_contract_t5': np.random.uniform(40000, 180000, 30),
            'weight': np.random.uniform(0.1, 1.0, 30)
        })

        with pytest.raises(ValueError) as exc_info:
            DataFrameValidator.validate_final_dataset(small_df, strict=True)
        assert "too small" in str(exc_info.value).lower()

    def test_short_date_range_warns(self, valid_final_dataset):
        """Date range < 365 days should issue a warning."""
        # valid_final_dataset has 100 days, so it should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataFrameValidator.validate_final_dataset(valid_final_dataset, strict=True)
            # Check if any warning mentions insufficient days
            date_warnings = [x for x in w if 'days' in str(x.message).lower()]
            assert len(date_warnings) > 0

    def test_high_mape_warns(self, valid_forecast_results):
        """High average MAPE (>50%) should issue a warning."""
        df_high_mape = valid_forecast_results.copy()
        df_high_mape['abs_pct_error'] = 0.6  # 60% MAPE

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataFrameValidator.validate_forecast_results(df_high_mape)
            mape_warnings = [x for x in w if 'mape' in str(x.message).lower()]
            assert len(mape_warnings) > 0

    def test_no_features_selected_fails(self, valid_feature_selection_results):
        """Feature selection with no features selected should fail."""
        df_no_selected = valid_feature_selection_results.copy()
        df_no_selected['selected'] = False  # No features selected

        with pytest.raises(ValueError) as exc_info:
            DataFrameValidator.validate_feature_selection_results(df_no_selected)
        assert "no features" in str(exc_info.value).lower()

    def test_many_features_selected_warns(self, valid_feature_selection_results):
        """More than 10 features selected should issue a warning."""
        # Create DataFrame with many selected features
        many_features = pd.DataFrame({
            'feature_name': [f'feature_{i}' for i in range(15)],
            'coefficient': np.random.randn(15),
            'p_value': np.random.uniform(0, 0.1, 15),
            'selected': [True] * 15
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataFrameValidator.validate_feature_selection_results(many_features)
            overfit_warnings = [x for x in w if 'overfitting' in str(x.message).lower()]
            assert len(overfit_warnings) > 0

    def test_strict_false_returns_original_on_error(self, valid_final_dataset):
        """Non-strict validation should return original DataFrame on error."""
        df_invalid = valid_final_dataset.copy()
        df_invalid.loc[0, 'sales'] = -1000  # Invalid value

        result = DataFrameValidator.validate_final_dataset(df_invalid, strict=False)
        # Should return original DataFrame, not raise
        assert len(result) == len(df_invalid)


# =============================================================================
# DATAFRAMEVALIDATOR METHOD TESTS
# =============================================================================


class TestDataFrameValidatorMethods:
    """Tests for DataFrameValidator class methods."""

    def test_sample_validation_mode(self, valid_final_dataset):
        """Sample validation should work for large datasets."""
        # Create larger dataset
        np.random.seed(42)
        large_df = pd.concat([valid_final_dataset] * 20, ignore_index=True)
        large_df['date'] = pd.date_range('2020-01-01', periods=len(large_df), freq='D')

        # Should pass with sample_validation=True
        result = DataFrameValidator.validate_final_dataset(
            large_df, strict=True, sample_validation=True
        )
        assert len(result) == len(large_df)

    def test_validate_and_log_to_mlflow_without_mlflow(self, valid_final_dataset):
        """validate_and_log_to_mlflow should work with log_to_mlflow=False."""
        result = DataFrameValidator.validate_and_log_to_mlflow(
            df=valid_final_dataset,
            schema=FINAL_DATASET_SCHEMA,
            schema_name="test_schema",
            validation_strict=False,
            log_to_mlflow=False
        )
        assert len(result) == len(valid_final_dataset)

    def test_batch_validate_with_mlflow(self, valid_final_dataset, valid_forecast_results):
        """batch_validate_with_mlflow should process multiple validations."""
        validations = [
            {
                'df': valid_final_dataset,
                'schema': FINAL_DATASET_SCHEMA,
                'schema_name': 'final_dataset',
                'log_to_mlflow': False
            },
            {
                'df': valid_forecast_results,
                'schema': FORECAST_RESULTS_SCHEMA,
                'schema_name': 'forecast_results',
                'log_to_mlflow': False
            }
        ]

        results = DataFrameValidator.batch_validate_with_mlflow(validations)

        assert results['total_validations'] == 2
        assert results['successful_validations'] == 2
        assert results['summary']['validation_success_rate'] == 1.0

    def test_batch_validate_handles_failures(self, valid_final_dataset):
        """batch_validate_with_mlflow should handle validation failures gracefully."""
        invalid_df = valid_final_dataset.copy()
        invalid_df.loc[0, 'sales'] = -1000  # Invalid

        validations = [
            {
                'df': valid_final_dataset,
                'schema': FINAL_DATASET_SCHEMA,
                'schema_name': 'valid_dataset',
                'log_to_mlflow': False
            },
            {
                'df': invalid_df,
                'schema': FINAL_DATASET_SCHEMA,
                'schema_name': 'invalid_dataset',
                'validation_strict': True,
                'log_to_mlflow': False
            }
        ]

        results = DataFrameValidator.batch_validate_with_mlflow(validations)

        assert results['total_validations'] == 2
        assert results['successful_validations'] == 1
        assert results['validation_results']['invalid_dataset']['status'] == 'FAILED'


# =============================================================================
# SCHEMAAWAREHDVCTRACKER TESTS
# =============================================================================


class TestSchemaAwareDVCTracker:
    """Tests for SchemaAwareDVCTracker class."""

    def test_validate_and_track_dataset_creates_file(
        self, valid_final_dataset, temp_dir
    ):
        """validate_and_track_dataset should save validated DataFrame to file."""
        output_path = str(temp_dir / "test_output.parquet")

        result = SchemaAwareDVCTracker.validate_and_track_dataset(
            df=valid_final_dataset,
            output_path=output_path,
            schema=FINAL_DATASET_SCHEMA,
            schema_name="test_dataset",
            validation_strict=False
        )

        assert result['validation_status'] == 'PASSED'
        assert Path(output_path).exists()

        # Verify saved data
        saved_df = pd.read_parquet(output_path)
        assert len(saved_df) == len(valid_final_dataset)

    def test_validate_and_track_collects_statistics(
        self, valid_final_dataset, temp_dir
    ):
        """validate_and_track_dataset should collect DataFrame statistics."""
        output_path = str(temp_dir / "test_stats.parquet")

        result = SchemaAwareDVCTracker.validate_and_track_dataset(
            df=valid_final_dataset,
            output_path=output_path,
            schema=FINAL_DATASET_SCHEMA,
            schema_name="stats_test",
            validation_strict=False
        )

        stats = result['statistics']
        assert 'shape' in stats
        assert 'columns_count' in stats
        assert 'null_values' in stats
        assert 'duplicate_rows' in stats
        assert 'memory_usage_mb' in stats
        assert 'date_range_days' in stats

    def test_bulk_validate_and_track(self, valid_final_dataset, valid_forecast_results, temp_dir):
        """bulk_validate_and_track should process multiple datasets."""
        datasets = [
            {
                'df': valid_final_dataset,
                'output_path': str(temp_dir / "bulk_1.parquet"),
                'schema': FINAL_DATASET_SCHEMA,
                'schema_name': 'bulk_test_1'
            },
            {
                'df': valid_forecast_results,
                'output_path': str(temp_dir / "bulk_2.parquet"),
                'schema': FORECAST_RESULTS_SCHEMA,
                'schema_name': 'bulk_test_2'
            }
        ]

        results = SchemaAwareDVCTracker.bulk_validate_and_track(datasets)

        assert results['total_datasets'] == 2
        assert results['successful_validations'] == 2
        assert Path(temp_dir / "bulk_1.parquet").exists()
        assert Path(temp_dir / "bulk_2.parquet").exists()

    def test_validation_failure_captured(self, valid_final_dataset, temp_dir):
        """Validation failure should be captured in results, not raise."""
        invalid_df = valid_final_dataset.copy()
        invalid_df.loc[0, 'sales'] = -1000

        output_path = str(temp_dir / "invalid.parquet")

        result = SchemaAwareDVCTracker.validate_and_track_dataset(
            df=invalid_df,
            output_path=output_path,
            schema=FINAL_DATASET_SCHEMA,
            schema_name="invalid_test",
            validation_strict=False  # Don't raise, just capture
        )

        # With strict=False, it saves original data even if invalid
        assert Path(output_path).exists()


# =============================================================================
# FILE OPERATIONS TESTS
# =============================================================================


class TestFileOperations:
    """Tests for file loading and saving operations."""

    def test_load_validated_dataset_from_file(self, valid_final_dataset, temp_dir):
        """load_validated_dataset should load and validate parquet files."""
        file_path = temp_dir / "test_load.parquet"
        valid_final_dataset.to_parquet(file_path)

        result = load_validated_dataset(str(file_path), FINAL_DATASET_SCHEMA)
        assert len(result) == len(valid_final_dataset)

    def test_load_validated_dataset_missing_file_raises(self, temp_dir):
        """load_validated_dataset should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_validated_dataset(str(temp_dir / "nonexistent.parquet"))

    def test_load_validated_dataset_invalid_data_raises(self, temp_dir):
        """load_validated_dataset should raise ValueError for invalid data."""
        invalid_df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=10, freq='D'),
            'sales': [-100] * 10  # Invalid: negative sales
        })
        file_path = temp_dir / "invalid.parquet"
        invalid_df.to_parquet(file_path)

        with pytest.raises(ValueError):
            load_validated_dataset(str(file_path), FINAL_DATASET_SCHEMA)

    def test_validate_and_save_dataset(self, valid_final_dataset, temp_dir):
        """validate_and_save_dataset should save validated data."""
        output_path = str(temp_dir / "saved.parquet")

        result = validate_and_save_dataset(
            df=valid_final_dataset,
            output_path=output_path,
            schema=FINAL_DATASET_SCHEMA,
            dvc_add=False  # Skip DVC tracking for test
        )

        assert Path(output_path).exists()
        assert len(result) == len(valid_final_dataset)

    def test_load_and_validate_with_dvc_context(self, valid_final_dataset, temp_dir):
        """load_and_validate_with_dvc_context should load files with DVC awareness."""
        file_path = temp_dir / "dvc_context.parquet"
        valid_final_dataset.to_parquet(file_path)

        result = load_and_validate_with_dvc_context(
            str(file_path), FINAL_DATASET_SCHEMA, "test_schema"
        )
        assert len(result) == len(valid_final_dataset)


# =============================================================================
# SCHEMA COERCION AND FILTERING TESTS
# =============================================================================


class TestSchemaCoercionAndFiltering:
    """Tests for schema coercion and strict filtering behavior."""

    def test_coercion_converts_types(self, valid_final_dataset):
        """Schema with coerce=True should convert compatible types."""
        df = valid_final_dataset.copy()
        # Convert sales to int (should be coerced to float)
        df['sales'] = df['sales'].astype(int)

        # Should pass due to coercion
        validated = FINAL_DATASET_SCHEMA.validate(df)
        assert validated['sales'].dtype == np.float64

    def test_strict_filter_removes_extra_columns(self, valid_final_dataset):
        """Schema with strict='filter' should remove unexpected columns."""
        df = valid_final_dataset.copy()
        df['unexpected_column'] = 'should_be_removed'
        df['another_extra'] = 123

        validated = FINAL_DATASET_SCHEMA.validate(df)

        # Extra columns should be filtered out
        assert 'unexpected_column' not in validated.columns
        assert 'another_extra' not in validated.columns

    def test_forecast_schema_allows_extra_columns(self, valid_forecast_results):
        """FORECAST_RESULTS_SCHEMA (strict=False) should allow extra columns."""
        df = valid_forecast_results.copy()
        df['ci_lower'] = df['y_predict'] * 0.9
        df['ci_upper'] = df['y_predict'] * 1.1

        validated = FORECAST_RESULTS_SCHEMA.validate(df)

        # Extra columns should be preserved
        assert 'ci_lower' in validated.columns
        assert 'ci_upper' in validated.columns


# =============================================================================
# MLflow INTEGRATION TESTS (with mocking)
# =============================================================================


class TestMLflowIntegration:
    """Tests for MLflow integration functionality."""

    def test_mlflow_unavailable_raises_when_requested(self, valid_final_dataset):
        """Should raise error when MLflow logging requested but unavailable."""
        with patch.dict('sys.modules', {'src.config.mlflow_config': None}):
            # When MLflow module can't be imported and logging is requested
            # Error may be RuntimeError or ValueError depending on strict mode
            with pytest.raises((RuntimeError, ValueError)) as exc_info:
                DataFrameValidator.validate_and_log_to_mlflow(
                    df=valid_final_dataset,
                    schema=FINAL_DATASET_SCHEMA,
                    schema_name="test",
                    validation_strict=True,
                    log_to_mlflow=True
                )
            assert "mlflow" in str(exc_info.value).lower()

    def test_final_dataset_with_mlflow_no_logging(self, valid_final_dataset):
        """validate_final_dataset_with_mlflow should work without MLflow logging."""
        result = DataFrameValidator.validate_final_dataset_with_mlflow(
            df=valid_final_dataset,
            strict=False,
            sample_validation=False,
            log_to_mlflow=False
        )
        assert len(result) == len(valid_final_dataset)

    def test_forecast_results_with_mlflow_no_logging(self, valid_forecast_results):
        """validate_forecast_results_with_mlflow should work without MLflow logging."""
        result = DataFrameValidator.validate_forecast_results_with_mlflow(
            df=valid_forecast_results,
            log_to_mlflow=False
        )
        assert len(result) == len(valid_forecast_results)
