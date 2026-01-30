"""
Tests for src/features/competitor_sign_utils.py

Comprehensive test coverage for functional competitor sign utilities:
- flip_competitor_signs: Single pattern sign flipping
- validate_competitor_columns: Column validation and identification
- flip_multiple_patterns: Multi-pattern composition
- create_flip_summary: Summary DataFrame generation

Target: 0% → 60%+ coverage
"""

import pytest
import pandas as pd
import numpy as np

from src.features.competitor_sign_utils import (
    flip_competitor_signs,
    validate_competitor_columns,
    flip_multiple_patterns,
    create_flip_summary,
    _validate_flip_summary_inputs,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_df():
    """Standard test DataFrame with competitor columns."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5, freq='W'),
        'sales': [100, 200, 300, 400, 500],
        'competitor_mid_t2': [1.5, 2.0, 2.5, 3.0, 3.5],
        'competitor_top5_t3': [3.0, 3.5, 4.0, 4.5, 5.0],
        'other_feature': [10, 20, 30, 40, 50],
    })


@pytest.fixture
def multi_pattern_df():
    """DataFrame with multiple column patterns."""
    return pd.DataFrame({
        'sales': [100, 200, 300],
        'competitor_mid_t2': [1.5, 2.0, 2.5],
        'competitor_top5_t3': [3.0, 3.5, 4.0],
        'rival_median_t1': [2.8, 2.6, 2.9],
        'rival_top3_t2': [4.0, 4.5, 5.0],
        'prudential_rate': [2.0, 2.1, 2.0],
    })


@pytest.fixture
def df_with_nans():
    """DataFrame with NaN values in competitor columns."""
    return pd.DataFrame({
        'sales': [100, 200, 300],
        'competitor_mid_t2': [1.5, np.nan, 2.5],
        'competitor_all_nan': [np.nan, np.nan, np.nan],
    })


@pytest.fixture
def df_non_numeric():
    """DataFrame with non-numeric competitor column."""
    return pd.DataFrame({
        'sales': [100, 200, 300],
        'competitor_mid_t2': [1.5, 2.0, 2.5],
        'competitor_text': ['a', 'b', 'c'],
    })


# =============================================================================
# Tests for flip_competitor_signs
# =============================================================================


class TestFlipCompetitorSigns:
    """Tests for flip_competitor_signs function."""

    def test_basic_sign_flip(self, sample_df):
        """1.1: Basic sign flipping works correctly."""
        result = flip_competitor_signs(sample_df, 'competitor_')

        # Original values should be negated
        assert result['competitor_mid_t2'].tolist() == [-1.5, -2.0, -2.5, -3.0, -3.5]
        assert result['competitor_top5_t3'].tolist() == [-3.0, -3.5, -4.0, -4.5, -5.0]

    def test_non_matching_columns_unchanged(self, sample_df):
        """1.2: Non-matching columns remain unchanged."""
        result = flip_competitor_signs(sample_df, 'competitor_')

        # Non-competitor columns should be unchanged
        assert result['sales'].tolist() == [100, 200, 300, 400, 500]
        assert result['other_feature'].tolist() == [10, 20, 30, 40, 50]

    def test_immutability(self, sample_df):
        """1.3: Original DataFrame is not modified."""
        original_values = sample_df['competitor_mid_t2'].tolist()
        _ = flip_competitor_signs(sample_df, 'competitor_')

        # Original should be unchanged
        assert sample_df['competitor_mid_t2'].tolist() == original_values

    def test_case_insensitive_matching(self, sample_df):
        """1.4: Pattern matching is case-insensitive."""
        result = flip_competitor_signs(sample_df, 'COMPETITOR_')
        assert result['competitor_mid_t2'].iloc[0] == -1.5

    def test_no_matching_columns_returns_copy(self, sample_df):
        """1.5: No matches returns unmodified copy."""
        result = flip_competitor_signs(sample_df, 'nonexistent_')

        # Should be a copy with same values
        pd.testing.assert_frame_equal(result, sample_df)
        assert result is not sample_df

    def test_partial_pattern_match(self, sample_df):
        """1.6: Partial pattern matching works."""
        result = flip_competitor_signs(sample_df, 'mid')

        # Only 'competitor_mid_t2' should be flipped
        assert result['competitor_mid_t2'].iloc[0] == -1.5
        assert result['competitor_top5_t3'].iloc[0] == 3.0  # Unchanged

    def test_handles_nan_values(self, df_with_nans):
        """1.7: NaN values are preserved during flip."""
        result = flip_competitor_signs(df_with_nans, 'competitor_')

        assert result['competitor_mid_t2'].iloc[0] == -1.5
        assert pd.isna(result['competitor_mid_t2'].iloc[1])
        assert result['competitor_mid_t2'].iloc[2] == -2.5

    def test_handles_all_nan_column(self, df_with_nans):
        """1.8: All-NaN columns are handled gracefully."""
        result = flip_competitor_signs(df_with_nans, 'competitor_')

        # All NaN column should still be all NaN
        assert result['competitor_all_nan'].isna().all()

    # Input validation tests
    def test_invalid_df_type_raises_typeerror(self):
        """1.9: Non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Expected pd.DataFrame"):
            flip_competitor_signs("not a dataframe", 'competitor_')

    def test_invalid_pattern_type_raises_typeerror(self, sample_df):
        """1.10: Non-string pattern raises TypeError."""
        with pytest.raises(TypeError, match="Expected str for column_pattern"):
            flip_competitor_signs(sample_df, 123)

    def test_empty_df_raises_valueerror(self):
        """1.11: Empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot process empty DataFrame"):
            flip_competitor_signs(empty_df, 'competitor_')

    def test_empty_pattern_raises_valueerror(self, sample_df):
        """1.12: Empty pattern string raises ValueError."""
        with pytest.raises(ValueError, match="column_pattern cannot be empty"):
            flip_competitor_signs(sample_df, '')

    def test_whitespace_pattern_raises_valueerror(self, sample_df):
        """1.13: Whitespace-only pattern raises ValueError."""
        with pytest.raises(ValueError, match="column_pattern cannot be empty"):
            flip_competitor_signs(sample_df, '   ')

    def test_negative_values_become_positive(self):
        """1.14: Negative values become positive after flip."""
        df = pd.DataFrame({
            'competitor_rate': [-1.5, -2.0, -2.5],
        })
        result = flip_competitor_signs(df, 'competitor_')

        assert result['competitor_rate'].tolist() == [1.5, 2.0, 2.5]

    def test_zero_values_unchanged(self):
        """1.15: Zero values remain zero."""
        df = pd.DataFrame({
            'competitor_rate': [0.0, 1.0, 0.0],
        })
        result = flip_competitor_signs(df, 'competitor_')

        assert result['competitor_rate'].iloc[0] == 0.0
        assert result['competitor_rate'].iloc[2] == 0.0


# =============================================================================
# Tests for validate_competitor_columns
# =============================================================================


class TestValidateCompetitorColumns:
    """Tests for validate_competitor_columns function."""

    def test_finds_matching_columns(self, sample_df):
        """2.1: Correctly identifies matching columns."""
        columns, is_valid = validate_competitor_columns(sample_df, 'competitor_')

        assert 'competitor_mid_t2' in columns
        assert 'competitor_top5_t3' in columns
        assert len(columns) == 2

    def test_valid_numeric_columns(self, sample_df):
        """2.2: Numeric columns pass validation."""
        columns, is_valid = validate_competitor_columns(sample_df, 'competitor_')

        assert is_valid is True

    def test_non_numeric_columns_invalid(self, df_non_numeric):
        """2.3: Non-numeric columns fail validation."""
        columns, is_valid = validate_competitor_columns(df_non_numeric, 'competitor_')

        assert 'competitor_text' in columns
        assert is_valid is False

    def test_all_nan_columns_invalid(self, df_with_nans):
        """2.4: All-NaN columns fail validation."""
        columns, is_valid = validate_competitor_columns(df_with_nans, 'competitor_')

        assert is_valid is False

    def test_no_matching_columns(self, sample_df):
        """2.5: No matches returns empty list and valid."""
        columns, is_valid = validate_competitor_columns(sample_df, 'nonexistent_')

        assert columns == []
        assert is_valid is True

    def test_case_insensitive_matching(self, sample_df):
        """2.6: Pattern matching is case-insensitive."""
        columns, _ = validate_competitor_columns(sample_df, 'COMPETITOR_')

        assert len(columns) == 2

    # Input validation tests
    def test_invalid_df_type_raises_typeerror(self):
        """2.7: Non-DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="Expected pd.DataFrame"):
            validate_competitor_columns([1, 2, 3], 'competitor_')

    def test_invalid_pattern_type_raises_typeerror(self, sample_df):
        """2.8: Non-string pattern raises TypeError."""
        with pytest.raises(TypeError, match="Expected str for column_pattern"):
            validate_competitor_columns(sample_df, ['competitor_'])

    def test_empty_df_raises_valueerror(self):
        """2.9: Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="Cannot validate empty DataFrame"):
            validate_competitor_columns(pd.DataFrame(), 'competitor_')

    def test_empty_pattern_raises_valueerror(self, sample_df):
        """2.10: Empty pattern raises ValueError."""
        with pytest.raises(ValueError, match="column_pattern cannot be empty"):
            validate_competitor_columns(sample_df, '')


# =============================================================================
# Tests for flip_multiple_patterns
# =============================================================================


class TestFlipMultiplePatterns:
    """Tests for flip_multiple_patterns function."""

    def test_single_pattern(self, multi_pattern_df):
        """3.1: Single pattern in list works."""
        result = flip_multiple_patterns(multi_pattern_df, ['competitor_'])

        assert result['competitor_mid_t2'].iloc[0] == -1.5
        assert result['rival_median_t1'].iloc[0] == 2.8  # Unchanged

    def test_multiple_patterns(self, multi_pattern_df):
        """3.2: Multiple patterns flip correct columns."""
        result = flip_multiple_patterns(multi_pattern_df, ['competitor_', 'rival_'])

        # Both patterns should be flipped
        assert result['competitor_mid_t2'].iloc[0] == -1.5
        assert result['rival_median_t1'].iloc[0] == -2.8

    def test_non_matching_patterns_ignored(self, multi_pattern_df):
        """3.3: Non-matching patterns are ignored."""
        result = flip_multiple_patterns(multi_pattern_df, ['competitor_', 'nonexistent_'])

        assert result['competitor_mid_t2'].iloc[0] == -1.5
        # Other columns unchanged
        assert result['prudential_rate'].iloc[0] == 2.0

    def test_immutability(self, multi_pattern_df):
        """3.4: Original DataFrame is not modified."""
        original_values = multi_pattern_df['competitor_mid_t2'].tolist()
        _ = flip_multiple_patterns(multi_pattern_df, ['competitor_', 'rival_'])

        assert multi_pattern_df['competitor_mid_t2'].tolist() == original_values

    def test_empty_string_patterns_skipped(self, multi_pattern_df):
        """3.5: Empty string patterns in list are skipped."""
        result = flip_multiple_patterns(multi_pattern_df, ['competitor_', '', 'rival_'])

        # Should still work with valid patterns
        assert result['competitor_mid_t2'].iloc[0] == -1.5
        assert result['rival_median_t1'].iloc[0] == -2.8

    def test_whitespace_patterns_skipped(self, multi_pattern_df):
        """3.6: Whitespace-only patterns are skipped."""
        result = flip_multiple_patterns(multi_pattern_df, ['competitor_', '   '])

        assert result['competitor_mid_t2'].iloc[0] == -1.5

    # Input validation tests
    def test_invalid_df_type_raises_typeerror(self):
        """3.7: Non-DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="Expected pd.DataFrame"):
            flip_multiple_patterns({'a': [1, 2]}, ['competitor_'])

    def test_invalid_patterns_type_raises_typeerror(self, multi_pattern_df):
        """3.8: Non-list patterns raises TypeError."""
        with pytest.raises(TypeError, match="Expected list for patterns"):
            flip_multiple_patterns(multi_pattern_df, 'competitor_')

    def test_empty_df_raises_valueerror(self):
        """3.9: Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="Cannot process empty DataFrame"):
            flip_multiple_patterns(pd.DataFrame(), ['competitor_'])

    def test_empty_patterns_list_raises_valueerror(self, multi_pattern_df):
        """3.10: Empty patterns list raises ValueError."""
        with pytest.raises(ValueError, match="patterns list cannot be empty"):
            flip_multiple_patterns(multi_pattern_df, [])

    def test_non_string_in_patterns_raises_typeerror(self, multi_pattern_df):
        """3.11: Non-string elements in patterns raises TypeError."""
        with pytest.raises(TypeError, match="All patterns must be strings"):
            flip_multiple_patterns(multi_pattern_df, ['competitor_', 123])


# =============================================================================
# Tests for _validate_flip_summary_inputs
# =============================================================================


class TestValidateFlipSummaryInputs:
    """Tests for _validate_flip_summary_inputs helper function."""

    def test_valid_inputs_pass(self, sample_df):
        """4.1: Valid inputs pass validation without exception."""
        flipped = flip_competitor_signs(sample_df, 'competitor_')
        # Should not raise
        _validate_flip_summary_inputs(sample_df, flipped, 'competitor_')

    def test_invalid_df_original_type(self, sample_df):
        """4.2: Non-DataFrame df_original raises TypeError."""
        with pytest.raises(TypeError, match="Expected pd.DataFrame for df_original"):
            _validate_flip_summary_inputs([1, 2], sample_df, 'competitor_')

    def test_invalid_df_flipped_type(self, sample_df):
        """4.3: Non-DataFrame df_flipped raises TypeError."""
        with pytest.raises(TypeError, match="Expected pd.DataFrame for df_flipped"):
            _validate_flip_summary_inputs(sample_df, "not df", 'competitor_')

    def test_invalid_pattern_type(self, sample_df):
        """4.4: Non-string pattern raises TypeError."""
        with pytest.raises(TypeError, match="Expected str for column_pattern"):
            _validate_flip_summary_inputs(sample_df, sample_df, 123)

    def test_empty_original_df_raises_valueerror(self, sample_df):
        """4.5: Empty original DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="Cannot create summary from empty DataFrames"):
            _validate_flip_summary_inputs(pd.DataFrame(), sample_df, 'competitor_')

    def test_empty_flipped_df_raises_valueerror(self, sample_df):
        """4.6: Empty flipped DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="Cannot create summary from empty DataFrames"):
            _validate_flip_summary_inputs(sample_df, pd.DataFrame(), 'competitor_')

    def test_mismatched_shapes_raises_valueerror(self, sample_df):
        """4.7: Mismatched shapes raises ValueError."""
        smaller_df = sample_df.head(3)
        with pytest.raises(ValueError, match="DataFrame shapes must match"):
            _validate_flip_summary_inputs(sample_df, smaller_df, 'competitor_')


# =============================================================================
# Tests for create_flip_summary
# =============================================================================


class TestCreateFlipSummary:
    """Tests for create_flip_summary function."""

    def test_summary_structure(self, sample_df):
        """5.1: Summary has correct columns."""
        flipped = flip_competitor_signs(sample_df, 'competitor_')
        summary = create_flip_summary(sample_df, flipped, 'competitor_')

        expected_cols = ['column_name', 'original_mean', 'flipped_mean', 'sign_changed', 'row_count']
        assert list(summary.columns) == expected_cols

    def test_summary_values(self, sample_df):
        """5.2: Summary values are correct."""
        flipped = flip_competitor_signs(sample_df, 'competitor_')
        summary = create_flip_summary(sample_df, flipped, 'competitor_')

        # Check competitor_mid_t2 row
        mid_row = summary[summary['column_name'] == 'competitor_mid_t2'].iloc[0]
        assert mid_row['original_mean'] == 2.5  # mean of [1.5, 2.0, 2.5, 3.0, 3.5]
        assert mid_row['flipped_mean'] == -2.5
        assert mid_row['sign_changed'] == True  # noqa: E712 - numpy bool comparison
        assert mid_row['row_count'] == 5

    def test_sign_changed_detection(self, sample_df):
        """5.3: Sign change is correctly detected."""
        flipped = flip_competitor_signs(sample_df, 'competitor_')
        summary = create_flip_summary(sample_df, flipped, 'competitor_')

        # Both columns should show sign change
        assert summary['sign_changed'].all()

    def test_no_matching_columns_empty_summary(self, sample_df):
        """5.4: No matches returns empty DataFrame with correct schema."""
        flipped = flip_competitor_signs(sample_df, 'nonexistent_')
        summary = create_flip_summary(sample_df, flipped, 'nonexistent_')

        assert len(summary) == 0
        assert list(summary.columns) == ['column_name', 'original_mean', 'flipped_mean', 'sign_changed', 'row_count']

    def test_summary_row_count(self, sample_df):
        """5.5: Row count matches original DataFrame."""
        flipped = flip_competitor_signs(sample_df, 'competitor_')
        summary = create_flip_summary(sample_df, flipped, 'competitor_')

        assert (summary['row_count'] == len(sample_df)).all()

    def test_summary_with_nan_values(self, df_with_nans):
        """5.6: Summary handles NaN values in means."""
        flipped = flip_competitor_signs(df_with_nans, 'competitor_mid')
        summary = create_flip_summary(df_with_nans, flipped, 'competitor_mid')

        # Mean should be computed ignoring NaN
        assert len(summary) == 1
        assert summary.iloc[0]['original_mean'] == 2.0  # mean of [1.5, NaN, 2.5] = 2.0

    def test_zero_mean_sign_change(self):
        """5.7: Zero mean has sign_changed = False (both positive zero)."""
        df = pd.DataFrame({
            'competitor_zero': [-1.0, 1.0],  # mean = 0
        })
        flipped = flip_competitor_signs(df, 'competitor_')
        summary = create_flip_summary(df, flipped, 'competitor_')

        # 0 == -0 so sign_changed should be False
        assert summary.iloc[0]['sign_changed'] == False  # noqa: E712 - numpy bool comparison


# =============================================================================
# Integration / Edge Case Tests
# =============================================================================


class TestIntegration:
    """Integration and edge case tests."""

    def test_full_workflow(self, sample_df):
        """6.1: Complete workflow from validation through summary."""
        # Validate
        columns, is_valid = validate_competitor_columns(sample_df, 'competitor_')
        assert is_valid

        # Flip
        flipped = flip_competitor_signs(sample_df, 'competitor_')

        # Summarize
        summary = create_flip_summary(sample_df, flipped, 'competitor_')

        # Verify end-to-end
        assert len(summary) == len(columns)
        assert summary['sign_changed'].all()

    def test_double_flip_restores_original(self, sample_df):
        """6.2: Flipping twice restores original values."""
        first_flip = flip_competitor_signs(sample_df, 'competitor_')
        second_flip = flip_competitor_signs(first_flip, 'competitor_')

        pd.testing.assert_frame_equal(
            second_flip[['competitor_mid_t2', 'competitor_top5_t3']],
            sample_df[['competitor_mid_t2', 'competitor_top5_t3']]
        )

    def test_large_dataframe(self):
        """6.3: Works with large DataFrames."""
        large_df = pd.DataFrame({
            'competitor_rate': np.random.randn(10000),
            'other_col': np.random.randn(10000),
        })

        result = flip_competitor_signs(large_df, 'competitor_')

        # Verify sign flip
        np.testing.assert_array_almost_equal(
            result['competitor_rate'].values,
            -large_df['competitor_rate'].values
        )

    def test_single_row_dataframe(self):
        """6.4: Works with single-row DataFrame."""
        single_row = pd.DataFrame({
            'competitor_rate': [5.0],
            'other': [10],
        })

        result = flip_competitor_signs(single_row, 'competitor_')
        assert result['competitor_rate'].iloc[0] == -5.0

    def test_many_competitor_columns(self):
        """6.5: Works with many competitor columns."""
        data = {f'competitor_col_{i}': [float(i)] * 3 for i in range(50)}
        data['other'] = [100, 200, 300]
        df = pd.DataFrame(data)

        result = flip_competitor_signs(df, 'competitor_')

        # All competitor columns should be negated
        for i in range(50):
            assert result[f'competitor_col_{i}'].iloc[0] == -float(i)

    def test_unicode_column_names(self):
        """6.6: Works with unicode in column names."""
        df = pd.DataFrame({
            'competitor_α': [1.0, 2.0],
            'competitor_β': [3.0, 4.0],
        })

        result = flip_competitor_signs(df, 'competitor_')
        assert result['competitor_α'].iloc[0] == -1.0
        assert result['competitor_β'].iloc[0] == -3.0
