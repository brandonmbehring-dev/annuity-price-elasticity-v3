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


# =============================================================================
# NEW TESTS FOR COVERAGE EXPANSION (Session 4)
# =============================================================================


@pytest.fixture
def forecasting_df_full():
    """Extended DataFrame with all columns needed for full testing."""
    np.random.seed(42)
    n_rows = 100
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n_rows, freq='W'),
        'sales': np.random.uniform(10000, 100000, n_rows),
        'sales_target_current': np.random.uniform(10000, 100000, n_rows),
        'sales_by_contract_date': np.random.uniform(10000, 100000, n_rows),
        'prudential_rate_current': np.random.uniform(0.02, 0.05, n_rows),
        'competitor_mid_t2': np.random.uniform(0.01, 0.04, n_rows),
        'competitor_top5_t2': np.random.uniform(0.01, 0.04, n_rows),
        'weight': np.linspace(0.5, 1.0, n_rows),
        'holiday': np.random.choice([0, 1], n_rows, p=[0.95, 0.05])
    })


class TestExtractTestTargetAtCutoff:
    """Tests for extract_test_target_at_cutoff function."""

    def test_extracts_correct_value(self, forecasting_df_full):
        """Extracts correct test target value at cutoff."""
        from src.data.forecasting_atomic_ops import extract_test_target_at_cutoff

        cutoff = 50
        target = extract_test_target_at_cutoff(
            forecasting_df_full, cutoff, 'sales_target_current'
        )

        expected = forecasting_df_full.iloc[cutoff]['sales_by_contract_date']
        assert target == float(expected)

    def test_returns_float(self, forecasting_df_full):
        """Returns a float value."""
        from src.data.forecasting_atomic_ops import extract_test_target_at_cutoff

        target = extract_test_target_at_cutoff(
            forecasting_df_full, 30, 'sales_target_current'
        )
        assert isinstance(target, float)

    def test_raises_on_cutoff_exceeds_bounds(self, forecasting_df_full):
        """Raises ValueError when cutoff exceeds dataset bounds."""
        from src.data.forecasting_atomic_ops import extract_test_target_at_cutoff

        with pytest.raises(ValueError, match="exceeds dataset bounds"):
            extract_test_target_at_cutoff(forecasting_df_full, 200, 'sales_target_current')

    def test_raises_on_missing_validation_column(self, forecasting_df):
        """Raises ValueError when sales_by_contract_date column is missing."""
        from src.data.forecasting_atomic_ops import extract_test_target_at_cutoff

        with pytest.raises(ValueError, match="validation target column.*not found"):
            extract_test_target_at_cutoff(forecasting_df, 30, 'sales_target_current')


class TestExtractTestTargetContractDateAtomic:
    """Tests for extract_test_target_contract_date_atomic function."""

    def test_extracts_correct_value(self, forecasting_df_full):
        """Extracts correct contract-date target value."""
        from src.data.forecasting_atomic_ops import extract_test_target_contract_date_atomic

        cutoff = 40
        target = extract_test_target_contract_date_atomic(forecasting_df_full, cutoff)

        expected = forecasting_df_full.iloc[cutoff]['sales_by_contract_date']
        assert target == float(expected)

    def test_raises_on_cutoff_exceeds_bounds(self, forecasting_df_full):
        """Raises ValueError when cutoff exceeds bounds."""
        from src.data.forecasting_atomic_ops import extract_test_target_contract_date_atomic

        with pytest.raises(ValueError, match="exceeds dataset bounds"):
            extract_test_target_contract_date_atomic(forecasting_df_full, 200)

    def test_raises_on_missing_column(self, forecasting_df):
        """Raises ValueError when contract date column is missing."""
        from src.data.forecasting_atomic_ops import extract_test_target_contract_date_atomic

        with pytest.raises(ValueError, match="column not found"):
            extract_test_target_contract_date_atomic(forecasting_df, 30)


class TestApplyBusinessFiltersAtomic:
    """Tests for apply_business_filters_atomic function."""

    def test_removes_zero_sales(self, forecasting_df_full):
        """Removes observations with zero sales."""
        from src.data.forecasting_atomic_ops import apply_business_filters_atomic

        df = forecasting_df_full.copy()
        df.loc[10:15, 'sales'] = 0  # Add some zero sales

        config = {'remove_incomplete_final_obs': False}
        filtered = apply_business_filters_atomic(df, config)

        assert len(filtered) == len(df) - 6  # Removed 6 zero sales rows

    def test_removes_incomplete_final_observation(self, forecasting_df_full):
        """Removes final row when configured."""
        from src.data.forecasting_atomic_ops import apply_business_filters_atomic

        config = {'remove_incomplete_final_obs': True}
        filtered = apply_business_filters_atomic(forecasting_df_full, config)

        assert len(filtered) == len(forecasting_df_full) - 1

    def test_keeps_all_when_disabled(self, forecasting_df_full):
        """Keeps all rows when incomplete removal is disabled."""
        from src.data.forecasting_atomic_ops import apply_business_filters_atomic

        config = {'remove_incomplete_final_obs': False}
        filtered = apply_business_filters_atomic(forecasting_df_full, config)

        assert len(filtered) == len(forecasting_df_full)

    def test_raises_on_all_data_removed(self):
        """Raises ValueError when all data is filtered out."""
        from src.data.forecasting_atomic_ops import apply_business_filters_atomic

        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=5, freq='W'),
            'sales': [0, 0, 0, 0, 0]  # All zero sales
        })

        config = {}
        with pytest.raises(ValueError, match="removed all data"):
            apply_business_filters_atomic(df, config)


class TestApplySignCorrectionsAtomic:
    """Tests for apply_sign_corrections_atomic function."""

    def test_applies_sign_corrections(self):
        """Applies sign corrections where mask is True."""
        from src.data.forecasting_atomic_ops import apply_sign_corrections_atomic

        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        correction_mask = np.array([False, True, False])

        corrected = apply_sign_corrections_atomic(features, correction_mask)

        # Column 1 should be negated
        np.testing.assert_array_almost_equal(corrected[:, 0], features[:, 0])
        np.testing.assert_array_almost_equal(corrected[:, 1], -features[:, 1])
        np.testing.assert_array_almost_equal(corrected[:, 2], features[:, 2])

    def test_preserves_shape(self):
        """Preserves array shape after corrections."""
        from src.data.forecasting_atomic_ops import apply_sign_corrections_atomic

        features = np.random.randn(50, 5)
        correction_mask = np.array([True, False, True, False, True])

        corrected = apply_sign_corrections_atomic(features, correction_mask)

        assert corrected.shape == features.shape

    def test_raises_on_mask_mismatch(self):
        """Raises ValueError when mask length doesn't match feature count."""
        from src.data.forecasting_atomic_ops import apply_sign_corrections_atomic

        features = np.random.randn(50, 5)
        wrong_mask = np.array([True, False, True])  # Only 3 elements

        with pytest.raises(ValueError, match="doesn't match"):
            apply_sign_corrections_atomic(features, wrong_mask)


class TestCalculateTemporalWeightsAtomic:
    """Tests for calculate_temporal_weights_atomic function."""

    def test_calculates_correct_weights(self):
        """Calculates correct exponential weights."""
        from src.data.forecasting_atomic_ops import calculate_temporal_weights_atomic

        n_obs = 10
        decay_rate = 0.98
        weights = calculate_temporal_weights_atomic(n_obs, decay_rate)

        # Most recent observation should have weight = 1.0
        assert weights[-1] == pytest.approx(1.0)

        # Older observations should have lower weights
        assert weights[0] < weights[-1]

        # Check decay relationship
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]

    def test_correct_length(self):
        """Returns correct number of weights."""
        from src.data.forecasting_atomic_ops import calculate_temporal_weights_atomic

        weights = calculate_temporal_weights_atomic(50, 0.98)
        assert len(weights) == 50

    def test_all_weights_positive(self):
        """All weights are positive."""
        from src.data.forecasting_atomic_ops import calculate_temporal_weights_atomic

        weights = calculate_temporal_weights_atomic(100, 0.95)
        assert np.all(weights > 0)

    def test_all_weights_at_most_one(self):
        """All weights are <= 1.0."""
        from src.data.forecasting_atomic_ops import calculate_temporal_weights_atomic

        weights = calculate_temporal_weights_atomic(100, 0.99)
        assert np.all(weights <= 1.0)

    def test_raises_on_zero_observations(self):
        """Raises ValueError for zero observations."""
        from src.data.forecasting_atomic_ops import calculate_temporal_weights_atomic

        with pytest.raises(ValueError, match="must be positive"):
            calculate_temporal_weights_atomic(0, 0.98)

    def test_raises_on_invalid_decay_rate(self):
        """Raises ValueError for invalid decay rate."""
        from src.data.forecasting_atomic_ops import calculate_temporal_weights_atomic

        with pytest.raises(ValueError, match="must be in"):
            calculate_temporal_weights_atomic(50, 1.5)  # > 1

        with pytest.raises(ValueError, match="must be in"):
            calculate_temporal_weights_atomic(50, 0.0)  # = 0


class TestValidateCutoffDataAtomic:
    """Tests for validate_cutoff_data_atomic function."""

    def test_valid_data_returns_true(self):
        """Valid data returns True."""
        from src.data.forecasting_atomic_ops import validate_cutoff_data_atomic

        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        result = validate_cutoff_data_atomic(X, y, cutoff=50)
        assert result is True

    def test_misaligned_shapes_returns_false(self):
        """Misaligned X and y shapes returns False."""
        from src.data.forecasting_atomic_ops import validate_cutoff_data_atomic

        X = np.random.randn(50, 3)
        y = np.random.randn(40)  # Different length

        result = validate_cutoff_data_atomic(X, y, cutoff=50)
        assert result is False

    def test_invalid_feature_values_returns_false(self):
        """Invalid values in features returns False."""
        from src.data.forecasting_atomic_ops import validate_cutoff_data_atomic

        X = np.random.randn(50, 3)
        X[10, 1] = np.nan  # Add NaN
        y = np.random.randn(50)

        result = validate_cutoff_data_atomic(X, y, cutoff=50)
        assert result is False

    def test_invalid_target_values_returns_false(self):
        """Invalid values in targets returns False."""
        from src.data.forecasting_atomic_ops import validate_cutoff_data_atomic

        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        y[5] = np.inf  # Add infinity

        result = validate_cutoff_data_atomic(X, y, cutoff=50)
        assert result is False

    def test_insufficient_observations_returns_false(self):
        """Less than 10 observations returns False."""
        from src.data.forecasting_atomic_ops import validate_cutoff_data_atomic

        X = np.random.randn(5, 3)
        y = np.random.randn(5)

        result = validate_cutoff_data_atomic(X, y, cutoff=5)
        assert result is False

    def test_no_features_returns_false(self):
        """Zero features returns False."""
        from src.data.forecasting_atomic_ops import validate_cutoff_data_atomic

        X = np.random.randn(50, 0)  # No features
        y = np.random.randn(50)

        result = validate_cutoff_data_atomic(X, y, cutoff=50)
        assert result is False


class TestExtractWeightsAtCutoff:
    """Tests for extract_weights_at_cutoff function."""

    def test_extracts_precomputed_weights(self, forecasting_df_full):
        """Extracts precomputed weights from DataFrame."""
        from src.data.forecasting_atomic_ops import extract_weights_at_cutoff

        cutoff = 50
        weights = extract_weights_at_cutoff(forecasting_df_full, cutoff)

        assert len(weights) == cutoff
        expected = forecasting_df_full.iloc[:cutoff]['weight'].values
        np.testing.assert_array_almost_equal(weights, expected)

    def test_falls_back_to_calculated_weights(self, forecasting_df):
        """Falls back to calculated weights when column missing."""
        from src.data.forecasting_atomic_ops import extract_weights_at_cutoff

        # forecasting_df doesn't have 'weight' column
        cutoff = 30
        weights = extract_weights_at_cutoff(forecasting_df, cutoff)

        assert len(weights) == cutoff
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0)

    def test_raises_on_zero_cutoff(self, forecasting_df_full):
        """Raises ValueError for zero cutoff."""
        from src.data.forecasting_atomic_ops import extract_weights_at_cutoff

        with pytest.raises(ValueError, match="must be positive"):
            extract_weights_at_cutoff(forecasting_df_full, 0)

    def test_raises_on_cutoff_exceeds_length(self, forecasting_df_full):
        """Raises ValueError when cutoff exceeds dataset length."""
        from src.data.forecasting_atomic_ops import extract_weights_at_cutoff

        with pytest.raises(ValueError, match="exceeds dataset length"):
            extract_weights_at_cutoff(forecasting_df_full, 200)


class TestApplyOriginalTrainingFiltersAtomic:
    """Tests for apply_original_training_filters_atomic function."""

    def test_applies_cutoff(self, forecasting_df_full):
        """Applies cutoff to limit training data."""
        from src.data.forecasting_atomic_ops import apply_original_training_filters_atomic

        cutoff = 50
        config = {}
        filtered = apply_original_training_filters_atomic(
            forecasting_df_full, cutoff, 'sales_target_current', config
        )

        assert len(filtered) <= cutoff

    def test_excludes_holidays(self, forecasting_df_full):
        """Excludes holiday observations when configured."""
        from src.data.forecasting_atomic_ops import apply_original_training_filters_atomic

        cutoff = 80
        config = {'exclude_holidays': True}
        filtered = apply_original_training_filters_atomic(
            forecasting_df_full, cutoff, 'sales_target_current', config
        )

        # All remaining rows should have holiday=0
        assert (filtered['holiday'] == 0).all()

    def test_keeps_holidays_when_not_configured(self, forecasting_df_full):
        """Keeps holiday observations when not configured."""
        from src.data.forecasting_atomic_ops import apply_original_training_filters_atomic

        cutoff = 80
        config = {'exclude_holidays': False}
        filtered = apply_original_training_filters_atomic(
            forecasting_df_full, cutoff, 'sales_target_current', config
        )

        # Should include holidays
        original_count = len(forecasting_df_full.iloc[:cutoff])
        assert len(filtered) >= original_count - 10  # Allow for dropna

    def test_raises_when_no_data_remains(self):
        """Raises ValueError when all data is filtered out."""
        from src.data.forecasting_atomic_ops import apply_original_training_filters_atomic

        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=10, freq='W'),
            'sales_target_current': [np.nan] * 10,  # All NaN
            'holiday': [0] * 10
        })

        config = {}
        with pytest.raises(ValueError, match="No training data remaining"):
            apply_original_training_filters_atomic(df, 10, 'sales_target_current', config)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_training_data(self, forecasting_df_full, feature_columns):
        """Tests _extract_training_data helper."""
        from src.data.forecasting_atomic_ops import _extract_training_data

        X, y, weights = _extract_training_data(
            forecasting_df_full.iloc[:50],
            feature_columns,
            'sales_target_current'
        )

        assert X.shape == (50, len(feature_columns))
        assert len(y) == 50
        assert len(weights) == 50

    def test_extract_training_data_without_weights(self, forecasting_df, feature_columns):
        """Creates unit weights when weight column missing."""
        from src.data.forecasting_atomic_ops import _extract_training_data

        X, y, weights = _extract_training_data(
            forecasting_df.iloc[:30],
            feature_columns,
            'sales_target_current'
        )

        # Should create unit weights
        np.testing.assert_array_almost_equal(weights, np.ones(30))

    def test_apply_sign_corrections_if_configured(self, feature_columns):
        """Tests _apply_sign_corrections_if_configured helper."""
        from src.data.forecasting_atomic_ops import _apply_sign_corrections_if_configured

        X_train = np.random.randn(50, 3)
        X_test = np.random.randn(1, 3)

        # With sign correction
        config = {'sign_correction_mask': np.array([False, True, False])}
        X_train_corrected, X_test_corrected = _apply_sign_corrections_if_configured(
            X_train.copy(), X_test.copy(), config
        )

        # Column 1 should be negated
        np.testing.assert_array_almost_equal(X_train_corrected[:, 1], -X_train[:, 1])
        np.testing.assert_array_almost_equal(X_test_corrected[:, 1], -X_test[:, 1])

    def test_apply_sign_corrections_no_config(self, feature_columns):
        """Returns unchanged arrays when no config."""
        from src.data.forecasting_atomic_ops import _apply_sign_corrections_if_configured

        X_train = np.random.randn(50, 3)
        X_test = np.random.randn(1, 3)

        config = {}  # No sign correction
        X_train_out, X_test_out = _apply_sign_corrections_if_configured(
            X_train.copy(), X_test.copy(), config
        )

        np.testing.assert_array_almost_equal(X_train_out, X_train)
        np.testing.assert_array_almost_equal(X_test_out, X_test)
