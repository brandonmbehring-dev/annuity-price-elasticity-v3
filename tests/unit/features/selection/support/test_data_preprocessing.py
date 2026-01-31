"""
Comprehensive tests for data preprocessing utilities.

Tests cover:
- Target transformation functions (log1p, log, sqrt)
- Autoregressive feature transformations
- Feature availability validation
- Feature subset preparation

Test organization follows CODING_STANDARDS.md Section 7.4.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any

from src.features.selection.support.data_preprocessing import (
    apply_target_transformation,
    apply_autoregressive_transforms,
    validate_feature_availability,
    prepare_feature_subset,
    _validate_target_for_transformation,
    _apply_transformation_math,
    _validate_base_features,
    _validate_candidate_features,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'target_variable': [100, 150, 200, 175, 225, 250, 300, 275, 325, 350],
        'base_feature_1': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        'base_feature_2': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
        'candidate_1': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        'candidate_2': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    })


@pytest.fixture
def autoregressive_dataframe() -> pd.DataFrame:
    """Create DataFrame with autoregressive features."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'sales_target_t1': [100, 150, 200, 250, 300],
        'sales_target_t2': [90, 140, 190, 240, 290],
        'sales_target_contract_t1': [80, 130, 180, 230, 280],
        'other_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def feature_selection_config() -> Dict[str, Any]:
    """Create sample feature selection configuration."""
    return {
        'target_variable': 'target_variable',
        'base_features': ['base_feature_1', 'base_feature_2'],
        'candidate_features': ['candidate_1', 'candidate_2'],
        'max_candidate_features': 10,
    }


# =============================================================================
# Tests: _validate_target_for_transformation
# =============================================================================


class TestValidateTargetForTransformation:
    """Tests for target validation helper."""

    def test_happy_path_returns_series(self, sample_dataframe):
        """Test returns target series when column exists."""
        result = _validate_target_for_transformation(
            sample_dataframe, 'target_variable'
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 10

    def test_error_missing_column(self, sample_dataframe):
        """Test raises ValueError for missing target."""
        with pytest.raises(ValueError, match="not found"):
            _validate_target_for_transformation(
                sample_dataframe, 'nonexistent_column'
            )


# =============================================================================
# Tests: _apply_transformation_math
# =============================================================================


class TestApplyTransformationMath:
    """Tests for mathematical transformation helper."""

    def test_log1p_transformation(self):
        """Test log1p transformation produces correct values."""
        values = pd.Series([0, 1, 9, 99])
        result, name = _apply_transformation_math(values, 'log1p', 'target')

        expected = np.log1p(values)
        pd.testing.assert_series_equal(result, expected)
        assert name == 'target_log1p'

    def test_log_transformation_positive_values(self):
        """Test log transformation with positive values."""
        values = pd.Series([1, 10, 100, 1000])
        result, name = _apply_transformation_math(values, 'log', 'target')

        expected = np.log(values)
        pd.testing.assert_series_equal(result, expected)
        assert name == 'target_log'

    def test_log_transformation_error_nonpositive(self):
        """Test log transformation raises error for non-positive values."""
        values = pd.Series([0, 1, 2, 3])
        with pytest.raises(ValueError, match="positive values"):
            _apply_transformation_math(values, 'log', 'target')

    def test_sqrt_transformation(self):
        """Test sqrt transformation produces correct values."""
        values = pd.Series([0, 1, 4, 9, 16])
        result, name = _apply_transformation_math(values, 'sqrt', 'target')

        expected = np.sqrt(values)
        pd.testing.assert_series_equal(result, expected)
        assert name == 'target_sqrt'

    def test_sqrt_transformation_error_negative(self):
        """Test sqrt transformation raises error for negative values."""
        values = pd.Series([-1, 0, 1, 4])
        with pytest.raises(ValueError, match="non-negative"):
            _apply_transformation_math(values, 'sqrt', 'target')

    def test_unsupported_transformation(self):
        """Test raises error for unsupported transformation type."""
        values = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unsupported transformation"):
            _apply_transformation_math(values, 'invalid', 'target')


# =============================================================================
# Tests: apply_target_transformation
# =============================================================================


class TestApplyTargetTransformation:
    """Tests for target transformation orchestration."""

    def test_no_transformation_returns_unchanged(self, sample_dataframe):
        """Test 'none' transformation returns original data."""
        result_df, result_name = apply_target_transformation(
            sample_dataframe, 'target_variable', 'none'
        )

        assert result_name == 'target_variable'
        pd.testing.assert_frame_equal(result_df, sample_dataframe)

    def test_log1p_adds_transformed_column(self, sample_dataframe):
        """Test log1p transformation adds new column."""
        result_df, result_name = apply_target_transformation(
            sample_dataframe, 'target_variable', 'log1p'
        )

        assert result_name == 'target_variable_log1p'
        assert 'target_variable_log1p' in result_df.columns

        # Verify transformation values
        expected = np.log1p(sample_dataframe['target_variable'])
        pd.testing.assert_series_equal(
            result_df['target_variable_log1p'],
            expected,
            check_names=False
        )

    def test_immutability_original_unchanged(self, sample_dataframe):
        """Test original DataFrame is not modified."""
        original_columns = list(sample_dataframe.columns)
        original_copy = sample_dataframe.copy()

        apply_target_transformation(
            sample_dataframe, 'target_variable', 'log1p'
        )

        assert list(sample_dataframe.columns) == original_columns
        pd.testing.assert_frame_equal(sample_dataframe, original_copy)

    def test_error_missing_target(self, sample_dataframe):
        """Test raises error for missing target variable."""
        with pytest.raises(ValueError, match="not found"):
            apply_target_transformation(
                sample_dataframe, 'nonexistent', 'log1p'
            )


# =============================================================================
# Tests: apply_autoregressive_transforms
# =============================================================================


class TestApplyAutoRegressiveTransforms:
    """Tests for autoregressive feature transformation."""

    def test_transforms_application_date_features(self, autoregressive_dataframe):
        """Test transforms sales_target_t* columns."""
        result = apply_autoregressive_transforms(autoregressive_dataframe)

        # Check log(1 + x) was applied
        original_t1 = autoregressive_dataframe['sales_target_t1']
        expected_t1 = np.log(1 + original_t1)
        pd.testing.assert_series_equal(
            result['sales_target_t1'],
            expected_t1,
            check_names=False
        )

    def test_transforms_contract_date_features(self, autoregressive_dataframe):
        """Test transforms sales_target_contract_t* columns."""
        result = apply_autoregressive_transforms(autoregressive_dataframe)

        original_contract = autoregressive_dataframe['sales_target_contract_t1']
        expected_contract = np.log(1 + original_contract)
        pd.testing.assert_series_equal(
            result['sales_target_contract_t1'],
            expected_contract,
            check_names=False
        )

    def test_does_not_transform_other_features(self, autoregressive_dataframe):
        """Test non-AR features are unchanged."""
        result = apply_autoregressive_transforms(autoregressive_dataframe)

        pd.testing.assert_series_equal(
            result['other_feature'],
            autoregressive_dataframe['other_feature'],
            check_names=False
        )

    def test_empty_dataframe_returns_empty(self):
        """Test empty DataFrame returns empty with warning."""
        empty_df = pd.DataFrame()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_autoregressive_transforms(empty_df)

            assert len(w) == 1
            assert "empty dataset" in str(w[0].message).lower()

        assert result.empty

    def test_no_ar_features_returns_unchanged(self, sample_dataframe):
        """Test DataFrame without AR features is unchanged."""
        result = apply_autoregressive_transforms(sample_dataframe)

        # Columns should match (data unchanged except potential copy)
        assert list(result.columns) == list(sample_dataframe.columns)

    def test_immutability(self, autoregressive_dataframe):
        """Test original DataFrame is not modified."""
        original_copy = autoregressive_dataframe.copy()

        apply_autoregressive_transforms(autoregressive_dataframe)

        pd.testing.assert_frame_equal(
            autoregressive_dataframe, original_copy
        )


# =============================================================================
# Tests: _validate_base_features
# =============================================================================


class TestValidateBaseFeatures:
    """Tests for base feature validation helper."""

    def test_happy_path_no_error(self, sample_dataframe):
        """Test passes when all base features exist."""
        # Should not raise
        _validate_base_features(
            sample_dataframe,
            ['base_feature_1', 'base_feature_2']
        )

    def test_error_missing_features(self, sample_dataframe):
        """Test raises error when base features missing."""
        with pytest.raises(ValueError, match="Required base features missing"):
            _validate_base_features(
                sample_dataframe,
                ['base_feature_1', 'missing_feature']
            )


# =============================================================================
# Tests: _validate_candidate_features
# =============================================================================


class TestValidateCandidateFeatures:
    """Tests for candidate feature validation helper."""

    def test_returns_available_and_missing(self, sample_dataframe):
        """Test returns correct available and missing lists."""
        available, missing = _validate_candidate_features(
            sample_dataframe,
            ['candidate_1', 'candidate_2', 'missing_candidate']
        )

        assert available == ['candidate_1', 'candidate_2']
        assert missing == ['missing_candidate']

    def test_error_no_candidates_available(self, sample_dataframe):
        """Test raises error when no candidates available."""
        with pytest.raises(ValueError, match="No candidate features available"):
            _validate_candidate_features(
                sample_dataframe,
                ['nonexistent_1', 'nonexistent_2']
            )


# =============================================================================
# Tests: validate_feature_availability
# =============================================================================


class TestValidateFeatureAvailability:
    """Tests for feature availability validation."""

    def test_happy_path_all_available(
        self, sample_dataframe, feature_selection_config
    ):
        """Test returns correct results when all features available."""
        available, missing = validate_feature_availability(
            sample_dataframe, feature_selection_config
        )

        assert available == ['candidate_1', 'candidate_2']
        assert missing == []

    def test_partial_candidates_available(
        self, sample_dataframe, feature_selection_config
    ):
        """Test handles partial candidate availability."""
        config = feature_selection_config.copy()
        config['candidate_features'] = ['candidate_1', 'missing_candidate']

        available, missing = validate_feature_availability(
            sample_dataframe, config
        )

        assert available == ['candidate_1']
        assert missing == ['missing_candidate']

    def test_error_empty_dataframe(self, feature_selection_config):
        """Test raises error for empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty dataset"):
            validate_feature_availability(empty_df, feature_selection_config)

    def test_error_missing_target(
        self, sample_dataframe, feature_selection_config
    ):
        """Test raises error when target variable missing."""
        config = feature_selection_config.copy()
        config['target_variable'] = 'nonexistent_target'

        with pytest.raises(ValueError, match="validation failed"):
            validate_feature_availability(sample_dataframe, config)

    def test_error_missing_base_features(
        self, sample_dataframe, feature_selection_config
    ):
        """Test raises error when base features missing."""
        config = feature_selection_config.copy()
        config['base_features'] = ['base_feature_1', 'missing_base']

        with pytest.raises(ValueError, match="validation failed"):
            validate_feature_availability(sample_dataframe, config)

    def test_warning_low_availability(self, sample_dataframe):
        """Test warns when candidate availability below 50%."""
        config = {
            'target_variable': 'target_variable',
            'base_features': ['base_feature_1'],
            'candidate_features': [
                'candidate_1', 'missing_1', 'missing_2', 'missing_3'
            ],
            'max_candidate_features': 10,
        }

        with pytest.warns(UserWarning, match="25.0%.*candidate features available"):
            validate_feature_availability(sample_dataframe, config)


# =============================================================================
# Tests: prepare_feature_subset
# =============================================================================


class TestPrepareFeatureSubset:
    """Tests for feature subset preparation."""

    def test_happy_path_returns_subset(self, sample_dataframe):
        """Test returns correct subset of features."""
        result = prepare_feature_subset(
            sample_dataframe,
            ['base_feature_1', 'candidate_1'],
            'target_variable'
        )

        assert list(result.columns) == [
            'base_feature_1', 'candidate_1', 'target_variable'
        ]
        assert len(result) == 10

    def test_error_missing_features(self, sample_dataframe):
        """Test raises error when requested features missing."""
        with pytest.raises(ValueError, match="Missing columns"):
            prepare_feature_subset(
                sample_dataframe,
                ['base_feature_1', 'nonexistent'],
                'target_variable'
            )

    def test_error_missing_target(self, sample_dataframe):
        """Test raises error when target missing."""
        with pytest.raises(ValueError, match="Missing columns"):
            prepare_feature_subset(
                sample_dataframe,
                ['base_feature_1'],
                'nonexistent_target'
            )

    def test_immutability(self, sample_dataframe):
        """Test original DataFrame is not modified."""
        original_copy = sample_dataframe.copy()

        prepare_feature_subset(
            sample_dataframe,
            ['base_feature_1'],
            'target_variable'
        )

        pd.testing.assert_frame_equal(sample_dataframe, original_copy)

    def test_returns_copy_not_view(self, sample_dataframe):
        """Test result is a copy, not a view."""
        result = prepare_feature_subset(
            sample_dataframe,
            ['base_feature_1'],
            'target_variable'
        )

        # Modifying result should not affect original
        result['base_feature_1'] = 999
        assert sample_dataframe['base_feature_1'].iloc[0] != 999


# =============================================================================
# Tests: Edge Cases and Mathematical Precision
# =============================================================================


class TestEdgeCasesAndPrecision:
    """Tests for edge cases and mathematical precision."""

    def test_log1p_handles_zero_values(self):
        """Test log1p correctly handles zero values."""
        df = pd.DataFrame({'target': [0, 0, 0, 1, 2]})
        result_df, _ = apply_target_transformation(df, 'target', 'log1p')

        # log1p(0) should be 0
        assert result_df['target_log1p'].iloc[0] == pytest.approx(0.0)

    def test_log1p_precision_small_values(self):
        """Test log1p maintains precision for small values."""
        # log1p is more precise than log(1+x) for small x
        df = pd.DataFrame({'target': [1e-10, 1e-8, 1e-6]})
        result_df, _ = apply_target_transformation(df, 'target', 'log1p')

        expected = np.log1p(df['target'])
        pd.testing.assert_series_equal(
            result_df['target_log1p'],
            expected,
            check_names=False,
            rtol=1e-12
        )

    def test_sqrt_precision(self):
        """Test sqrt maintains precision."""
        df = pd.DataFrame({'target': [1.0, 2.0, 3.0, 4.0]})
        result_df, _ = apply_target_transformation(df, 'target', 'sqrt')

        expected = np.sqrt(df['target'])
        pd.testing.assert_series_equal(
            result_df['target_sqrt'],
            expected,
            check_names=False,
            rtol=1e-12
        )

    def test_ar_transform_large_values(self):
        """Test AR transform handles large values correctly."""
        df = pd.DataFrame({
            'sales_target_t1': [1e6, 1e7, 1e8],
        })

        result = apply_autoregressive_transforms(df)
        expected = np.log(1 + df['sales_target_t1'])

        pd.testing.assert_series_equal(
            result['sales_target_t1'],
            expected,
            check_names=False,
            rtol=1e-12
        )

    def test_single_row_dataframe(self, feature_selection_config):
        """Test functions handle single-row DataFrame."""
        single_row = pd.DataFrame({
            'target_variable': [100],
            'base_feature_1': [1.0],
            'base_feature_2': [2.0],
            'candidate_1': [0.5],
            'candidate_2': [0.3],
        })

        # Should not raise
        available, missing = validate_feature_availability(
            single_row, feature_selection_config
        )
        assert len(available) == 2


# =============================================================================
# Tests: _load_and_log_dataset
# =============================================================================


class TestLoadAndLogDataset:
    """Tests for _load_and_log_dataset function."""

    def test_loads_parquet_file(self, tmp_path):
        """Test successfully loads parquet file."""
        from src.features.selection.support.data_preprocessing import _load_and_log_dataset

        # Create test parquet file
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_path = tmp_path / 'test_data.parquet'
        test_df.to_parquet(test_path)

        result = _load_and_log_dataset(str(test_path))

        pd.testing.assert_frame_equal(result, test_df)

    def test_raises_on_missing_file(self):
        """Test raises ValueError for missing file."""
        from src.features.selection.support.data_preprocessing import _load_and_log_dataset

        with pytest.raises(ValueError, match="Dataset file not found"):
            _load_and_log_dataset('/nonexistent/path/file.parquet')


# =============================================================================
# Tests: _apply_temporal_filters
# =============================================================================


class TestApplyTemporalFilters:
    """Tests for _apply_temporal_filters function."""

    def test_applies_date_filter(self):
        """Test applies date filtering correctly."""
        from src.features.selection.support.data_preprocessing import _apply_temporal_filters

        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-06-01', '2022-08-15', '2022-09-01', '2022-10-01']),
            'value': [1, 2, 3, 4]
        })
        config = {
            'analysis_start_date': '2022-08-01',
            'exclude_holidays': False,
            'target_variable': 'value',
            'base_features': [],
            'candidate_features': [],
            'max_candidate_features': 10
        }

        result = _apply_temporal_filters(data, config)

        # Should only include rows after 2022-08-01
        assert len(result) == 3
        assert result['date'].min() > pd.Timestamp('2022-08-01')

    def test_applies_holiday_filter(self):
        """Test applies holiday filtering when enabled."""
        from src.features.selection.support.data_preprocessing import _apply_temporal_filters

        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-09-01', '2022-09-02', '2022-09-03', '2022-09-04']),
            'holiday': [0, 1, 0, 1],
            'value': [1, 2, 3, 4]
        })
        config = {
            'analysis_start_date': '2022-08-01',
            'exclude_holidays': True,
            'target_variable': 'value',
            'base_features': [],
            'candidate_features': [],
            'max_candidate_features': 10
        }

        result = _apply_temporal_filters(data, config)

        # Should only include non-holiday rows
        assert len(result) == 2
        assert (result['holiday'] == 0).all()

    def test_skips_holiday_filter_when_disabled(self):
        """Test skips holiday filter when exclude_holidays is False."""
        from src.features.selection.support.data_preprocessing import _apply_temporal_filters

        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-09-01', '2022-09-02']),
            'holiday': [0, 1],
            'value': [1, 2]
        })
        config = {
            'analysis_start_date': '2022-08-01',
            'exclude_holidays': False,
            'target_variable': 'value',
            'base_features': [],
            'candidate_features': [],
            'max_candidate_features': 10
        }

        result = _apply_temporal_filters(data, config)

        # Should include all rows
        assert len(result) == 2

    def test_uses_default_start_date(self):
        """Test uses default analysis start date when not provided."""
        from src.features.selection.support.data_preprocessing import _apply_temporal_filters

        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-06-01', '2022-08-15', '2022-09-01']),
            'value': [1, 2, 3]
        })
        config = {
            # No analysis_start_date specified
            'exclude_holidays': False,
            'target_variable': 'value',
            'base_features': [],
            'candidate_features': [],
            'max_candidate_features': 10
        }

        result = _apply_temporal_filters(data, config)

        # Default is '2022-08-01', so should include only 2 rows
        assert len(result) == 2


# =============================================================================
# Tests: _apply_configured_transformations
# =============================================================================


class TestApplyConfiguredTransformations:
    """Tests for _apply_configured_transformations function."""

    def test_no_transformation(self):
        """Test with no target transformation configured."""
        from src.features.selection.support.data_preprocessing import _apply_configured_transformations

        data = pd.DataFrame({
            'target': [100, 200, 300],
            'feature': [1, 2, 3]
        })
        config = {
            'target_variable': 'target',
            'target_transformation': 'none',
            'base_features': [],
            'candidate_features': [],
            'max_candidate_features': 10
        }

        result_df, effective_target = _apply_configured_transformations(data, config)

        assert effective_target == 'target'
        assert 'target' in result_df.columns

    def test_with_log1p_transformation(self):
        """Test with log1p target transformation."""
        from src.features.selection.support.data_preprocessing import _apply_configured_transformations

        data = pd.DataFrame({
            'target': [100, 200, 300],
            'feature': [1, 2, 3]
        })
        config = {
            'target_variable': 'target',
            'target_transformation': 'log1p',
            'transformed_target_suffix': '_transformed',
            'base_features': [],
            'candidate_features': [],
            'max_candidate_features': 10
        }

        result_df, effective_target = _apply_configured_transformations(data, config)

        assert effective_target == 'target_log1p'
        assert 'target_log1p' in result_df.columns

    def test_default_transformation_is_none(self):
        """Test that default transformation is 'none' when not specified."""
        from src.features.selection.support.data_preprocessing import _apply_configured_transformations

        data = pd.DataFrame({
            'target': [100, 200, 300],
            'feature': [1, 2, 3]
        })
        config = {
            'target_variable': 'target',
            # No target_transformation specified
            'base_features': [],
            'candidate_features': [],
            'max_candidate_features': 10
        }

        result_df, effective_target = _apply_configured_transformations(data, config)

        assert effective_target == 'target'


# =============================================================================
# Tests: prepare_analysis_dataset
# =============================================================================


class TestPrepareAnalysisDataset:
    """Tests for prepare_analysis_dataset function."""

    def test_full_pipeline(self, tmp_path):
        """Test full dataset preparation pipeline."""
        from src.features.selection.support.data_preprocessing import prepare_analysis_dataset

        # Create test parquet file
        test_df = pd.DataFrame({
            'date': pd.to_datetime(['2022-09-01', '2022-09-08', '2022-09-15', '2022-09-22']),
            'target_var': [100, 150, 200, 250],
            'base_feat': [1.0, 1.1, 1.2, 1.3],
            'cand_feat': [0.5, 0.6, 0.7, 0.8],
            'holiday': [0, 0, 0, 0]
        })
        test_path = tmp_path / 'test_data.parquet'
        test_df.to_parquet(test_path)

        config = {
            'target_variable': 'target_var',
            'base_features': ['base_feat'],
            'candidate_features': ['cand_feat'],
            'max_candidate_features': 10,
            'analysis_start_date': '2022-08-01',
            'exclude_holidays': True
        }

        result_df, effective_target = prepare_analysis_dataset(str(test_path), config)

        assert len(result_df) == 4
        assert effective_target == 'target_var'
        assert 'base_feat' in result_df.columns
        assert 'cand_feat' in result_df.columns

    def test_raises_on_invalid_config(self, tmp_path):
        """Test raises ValueError on invalid configuration."""
        from src.features.selection.support.data_preprocessing import prepare_analysis_dataset

        # Create test parquet file
        test_df = pd.DataFrame({
            'date': pd.to_datetime(['2022-09-01']),
            'value': [100]
        })
        test_path = tmp_path / 'test_data.parquet'
        test_df.to_parquet(test_path)

        config = {
            'target_variable': 'nonexistent_target',
            'base_features': ['also_nonexistent'],
            'candidate_features': [],
            'max_candidate_features': 10,
            'analysis_start_date': '2022-08-01'
        }

        with pytest.raises(ValueError, match="preparation failed"):
            prepare_analysis_dataset(str(test_path), config)


# =============================================================================
# Tests: prepare_feature_subset (additional coverage)
# =============================================================================


class TestPrepareFeatureSubsetAdditional:
    """Additional tests for prepare_feature_subset function."""

    def test_warns_on_missing_values(self, sample_dataframe, capsys):
        """Test that missing values trigger a warning message."""
        from src.features.selection.support.data_preprocessing import prepare_feature_subset

        # Add missing values to sample data
        df_with_nan = sample_dataframe.copy()
        df_with_nan.loc[0, 'candidate_1'] = np.nan
        df_with_nan.loc[1, 'candidate_1'] = np.nan

        result = prepare_feature_subset(
            df_with_nan,
            ['candidate_1'],
            'target_variable'
        )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "missing values" in captured.out.lower()
        assert len(result) == 10


# =============================================================================
# Tests: Exception handling in apply_autoregressive_transforms
# =============================================================================


class TestApplyAutoRegressiveTransformsExceptionHandling:
    """Tests for exception handling in apply_autoregressive_transforms."""

    def test_handles_transformation_error_gracefully(self):
        """Test that transformation errors are caught and warned."""
        from src.features.selection.support.data_preprocessing import apply_autoregressive_transforms

        # Create a dataframe where transformation might fail
        # (e.g., object dtype that can't be log-transformed)
        df = pd.DataFrame({
            'sales_target_t1': ['a', 'b', 'c'],  # String values
            'other': [1, 2, 3]
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_autoregressive_transforms(df)

            # Should return copy and issue warning
            assert isinstance(result, pd.DataFrame)
