"""
Unit tests for src/data/forecasting_atomic_ops.py

Tests atomic forecasting operations including feature extraction,
target extraction, and cutoff validation.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def forecasting_df():
    """DataFrame for forecasting atomic operations testing."""
    np.random.seed(42)
    n_rows = 100
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n_rows, freq='W'),
        'sales_target_current': np.random.uniform(10000, 100000, n_rows),
        'prudential_rate_current': np.random.uniform(0.02, 0.05, n_rows),
        'competitor_mid_t2': np.random.uniform(0.01, 0.04, n_rows),
        'competitor_top5_t2': np.random.uniform(0.01, 0.04, n_rows),
    })


@pytest.fixture
def feature_columns():
    """Standard feature columns for testing."""
    return ['prudential_rate_current', 'competitor_mid_t2', 'competitor_top5_t2']


class TestValidateCutoffBounds:
    """Tests for _validate_cutoff_bounds function."""

    def test_valid_cutoff_passes(self, forecasting_df):
        """Valid cutoff within bounds passes validation."""
        from src.data.forecasting_atomic_ops import _validate_cutoff_bounds

        # Should not raise
        _validate_cutoff_bounds(50, len(forecasting_df))

    def test_zero_cutoff_raises(self, forecasting_df):
        """Zero cutoff raises ValueError."""
        from src.data.forecasting_atomic_ops import _validate_cutoff_bounds

        with pytest.raises(ValueError, match="must be positive"):
            _validate_cutoff_bounds(0, len(forecasting_df))

    def test_negative_cutoff_raises(self, forecasting_df):
        """Negative cutoff raises ValueError."""
        from src.data.forecasting_atomic_ops import _validate_cutoff_bounds

        with pytest.raises(ValueError, match="must be positive"):
            _validate_cutoff_bounds(-5, len(forecasting_df))

    def test_cutoff_exceeds_length_raises(self, forecasting_df):
        """Cutoff exceeding dataset length raises ValueError."""
        from src.data.forecasting_atomic_ops import _validate_cutoff_bounds

        with pytest.raises(ValueError, match="exceeds dataset length"):
            _validate_cutoff_bounds(200, len(forecasting_df))


class TestValidateFeatureColumnsExist:
    """Tests for _validate_feature_columns_exist function."""

    def test_valid_columns_pass(self, forecasting_df, feature_columns):
        """Valid feature columns pass validation."""
        from src.data.forecasting_atomic_ops import _validate_feature_columns_exist

        # Should not raise
        _validate_feature_columns_exist(forecasting_df, feature_columns)

    def test_missing_columns_raise(self, forecasting_df):
        """Missing feature columns raise ValueError."""
        from src.data.forecasting_atomic_ops import _validate_feature_columns_exist

        with pytest.raises(ValueError, match="missing features"):
            _validate_feature_columns_exist(forecasting_df, ['nonexistent_column'])

    def test_partial_missing_columns_raise(self, forecasting_df):
        """Partially missing columns raise ValueError."""
        from src.data.forecasting_atomic_ops import _validate_feature_columns_exist

        columns = ['prudential_rate_current', 'missing_column']
        with pytest.raises(ValueError, match="missing features"):
            _validate_feature_columns_exist(forecasting_df, columns)


class TestExtractFeaturesAtCutoff:
    """Tests for extract_features_at_cutoff function."""

    def test_extracts_correct_shape(self, forecasting_df, feature_columns):
        """Extracts feature matrix with correct shape."""
        from src.data.forecasting_atomic_ops import extract_features_at_cutoff

        cutoff = 50
        features = extract_features_at_cutoff(forecasting_df, cutoff, feature_columns)

        assert features.shape == (cutoff, len(feature_columns))

    def test_extracts_correct_values(self, forecasting_df, feature_columns):
        """Extracts correct feature values."""
        from src.data.forecasting_atomic_ops import extract_features_at_cutoff

        cutoff = 10
        features = extract_features_at_cutoff(forecasting_df, cutoff, feature_columns)

        # Check first column matches source data
        expected = forecasting_df.iloc[:cutoff][feature_columns[0]].values
        np.testing.assert_array_almost_equal(features[:, 0], expected)

    def test_returns_numpy_array(self, forecasting_df, feature_columns):
        """Returns numpy ndarray."""
        from src.data.forecasting_atomic_ops import extract_features_at_cutoff

        features = extract_features_at_cutoff(forecasting_df, 30, feature_columns)
        assert isinstance(features, np.ndarray)

    def test_no_data_leakage(self, forecasting_df, feature_columns):
        """Ensures no data beyond cutoff is included."""
        from src.data.forecasting_atomic_ops import extract_features_at_cutoff

        cutoff = 25
        features = extract_features_at_cutoff(forecasting_df, cutoff, feature_columns)

        # Should only have cutoff rows
        assert features.shape[0] == cutoff


class TestValidateTargetColumnExists:
    """Tests for _validate_target_column_exists function."""

    def test_valid_target_passes(self, forecasting_df):
        """Valid target column passes validation."""
        from src.data.forecasting_atomic_ops import _validate_target_column_exists

        # Should not raise
        _validate_target_column_exists(forecasting_df, 'sales_target_current')

    def test_missing_target_raises(self, forecasting_df):
        """Missing target column raises ValueError."""
        from src.data.forecasting_atomic_ops import _validate_target_column_exists

        with pytest.raises(ValueError, match="target column.*not found"):
            _validate_target_column_exists(forecasting_df, 'nonexistent_target')


class TestExtractTargetAtCutoff:
    """Tests for extract_target_at_cutoff function."""

    def test_extracts_correct_length(self, forecasting_df):
        """Extracts target values with correct length."""
        from src.data.forecasting_atomic_ops import extract_target_at_cutoff

        cutoff = 40
        target = extract_target_at_cutoff(
            forecasting_df, cutoff, 'sales_target_current'
        )

        assert len(target) == cutoff

    def test_extracts_correct_values(self, forecasting_df):
        """Extracts correct target values."""
        from src.data.forecasting_atomic_ops import extract_target_at_cutoff

        cutoff = 15
        target = extract_target_at_cutoff(
            forecasting_df, cutoff, 'sales_target_current'
        )

        expected = forecasting_df.iloc[:cutoff]['sales_target_current'].values
        np.testing.assert_array_almost_equal(target, expected)

    def test_returns_numpy_array(self, forecasting_df):
        """Returns numpy ndarray."""
        from src.data.forecasting_atomic_ops import extract_target_at_cutoff

        target = extract_target_at_cutoff(
            forecasting_df, 30, 'sales_target_current'
        )
        assert isinstance(target, np.ndarray)

    def test_1d_array_shape(self, forecasting_df):
        """Returns 1D array."""
        from src.data.forecasting_atomic_ops import extract_target_at_cutoff

        target = extract_target_at_cutoff(
            forecasting_df, 30, 'sales_target_current'
        )
        assert target.ndim == 1


class TestExtractTestFeaturesAtCutoff:
    """Tests for extract_test_features_at_cutoff function."""

    def test_extracts_test_features(self, forecasting_df, feature_columns):
        """Extracts test features at cutoff point."""
        from src.data.forecasting_atomic_ops import extract_test_features_at_cutoff

        cutoff = 50
        test_features = extract_test_features_at_cutoff(
            forecasting_df, cutoff, feature_columns
        )

        # Test features should be for the observation at cutoff
        assert test_features.shape == (1, len(feature_columns))

    def test_extracts_correct_test_values(self, forecasting_df, feature_columns):
        """Extracts correct test feature values at cutoff."""
        from src.data.forecasting_atomic_ops import extract_test_features_at_cutoff

        cutoff = 30
        test_features = extract_test_features_at_cutoff(
            forecasting_df, cutoff, feature_columns
        )

        # Should match the row at cutoff index
        expected = forecasting_df.iloc[cutoff][feature_columns].values
        np.testing.assert_array_almost_equal(test_features.flatten(), expected)


class TestAtomicOperationsIntegration:
    """Integration tests for atomic operations."""

    def test_feature_target_extraction_consistent(self, forecasting_df, feature_columns):
        """Feature and target extraction are consistent."""
        from src.data.forecasting_atomic_ops import (
            extract_features_at_cutoff,
            extract_target_at_cutoff
        )

        cutoff = 45
        features = extract_features_at_cutoff(forecasting_df, cutoff, feature_columns)
        target = extract_target_at_cutoff(
            forecasting_df, cutoff, 'sales_target_current'
        )

        # Same number of observations
        assert features.shape[0] == len(target)

    def test_expanding_window_simulation(self, forecasting_df, feature_columns):
        """Simulates expanding window cross-validation extraction."""
        from src.data.forecasting_atomic_ops import (
            extract_features_at_cutoff,
            extract_target_at_cutoff,
            extract_test_features_at_cutoff
        )

        results = []
        for cutoff in range(30, 60, 5):
            X_train = extract_features_at_cutoff(forecasting_df, cutoff, feature_columns)
            y_train = extract_target_at_cutoff(
                forecasting_df, cutoff, 'sales_target_current'
            )
            X_test = extract_test_features_at_cutoff(forecasting_df, cutoff, feature_columns)

            results.append({
                'cutoff': cutoff,
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
            })

        # Training size should increase with cutoff
        assert results[0]['train_size'] < results[-1]['train_size']
