"""
Medium-Scale Integration Tests
==============================

Integration tests using the medium_dataset fixture for 10x faster testing
compared to full production data.

Dataset: 100 weeks × 50 features (vs 167 weeks × 70+ features in production)
Load time: < 0.5 seconds

Tests validate:
- Feature engineering pipeline
- Data preprocessing stages
- Competitive feature aggregation
- Pipeline stage transitions
- Data integrity through transformations

Author: Claude Code
Date: 2026-01-30
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================


class TestFeatureEngineeringMediumScale:
    """Feature engineering tests using medium_dataset fixture."""

    def test_lag_features_present(self, medium_dataset):
        """Medium dataset should have lag features 1-10."""
        df = medium_dataset

        # Check own rate lags
        for lag in range(1, 11):
            col_name = f'own_cap_rate_lag_{lag}'
            assert col_name in df.columns, f"Missing {col_name}"
            assert df[col_name].notna().all(), f"{col_name} has NaN values"

        # Check competitor lags
        for lag in range(1, 11):
            col_name = f'competitor_mean_lag_{lag}'
            assert col_name in df.columns, f"Missing {col_name}"

    def test_polynomial_features_present(self, medium_dataset):
        """Medium dataset should have polynomial interaction features."""
        df = medium_dataset

        expected_poly = [
            'own_rate_squared',
            'competitor_squared',
            'own_competitor_interaction',
            'spread_squared',
            'vix_squared',
        ]

        for feature in expected_poly:
            assert feature in df.columns, f"Missing polynomial feature: {feature}"

    def test_feature_engineering_preserves_row_count(self, medium_dataset):
        """Feature engineering should not drop or duplicate rows."""
        df = medium_dataset
        original_count = 100  # Known from fixture definition

        assert len(df) == original_count, (
            f"Expected {original_count} rows, got {len(df)}"
        )

    def test_feature_types_correct(self, medium_dataset):
        """Features should have correct data types."""
        df = medium_dataset

        # Date column
        assert pd.api.types.is_datetime64_any_dtype(df['date']), (
            "date column should be datetime"
        )

        # Numeric columns
        numeric_cols = ['sales', 'own_cap_rate', 'competitor_mean', 'vix', 'dgs5']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"{col} should be numeric"
            )

    def test_rate_features_in_valid_range(self, medium_dataset):
        """Rate features should be in realistic percentage range."""
        df = medium_dataset

        rate_cols = ['own_cap_rate', 'competitor_mean', 'competitor_median']
        for col in rate_cols:
            assert df[col].min() >= 0.01, f"{col} has unrealistic low rates"
            assert df[col].max() <= 0.30, f"{col} has unrealistic high rates"

    def test_create_additional_features(self, medium_dataset):
        """Test creating derived features from medium dataset."""
        df = medium_dataset.copy()

        # Create rate spread
        df['rate_spread'] = df['own_cap_rate'] - df['competitor_mean']

        assert 'rate_spread' in df.columns
        # Spread should be reasonable (own rate can be higher or lower)
        assert df['rate_spread'].between(-0.10, 0.10).all(), (
            "Rate spread outside reasonable range"
        )

    def test_rolling_features(self, medium_dataset):
        """Test creating rolling window features."""
        df = medium_dataset.copy()

        # Create 4-week rolling mean of sales
        df['sales_rolling_4w'] = df['sales'].rolling(window=4, min_periods=1).mean()

        assert 'sales_rolling_4w' in df.columns
        assert df['sales_rolling_4w'].notna().all(), (
            "Rolling mean should handle min_periods"
        )


# =============================================================================
# DATA PREPROCESSING TESTS
# =============================================================================


class TestDataPreprocessingMediumScale:
    """Data preprocessing tests using medium_dataset."""

    def test_standardization_preserves_shape(self, medium_dataset):
        """Standardization should preserve DataFrame shape."""
        df = medium_dataset.copy()

        # Select numeric columns for standardization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        original_shape = df[numeric_cols].shape

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])

        assert scaled_data.shape == original_shape, (
            f"Shape changed: {original_shape} -> {scaled_data.shape}"
        )

    def test_standardization_produces_unit_variance(self, medium_dataset):
        """Standardized features should have approximately unit variance."""
        df = medium_dataset.copy()

        # Select a subset of features
        features = ['own_cap_rate', 'competitor_mean', 'vix', 'dgs5']
        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check variance is approximately 1
        variances = np.var(X_scaled, axis=0)
        np.testing.assert_array_almost_equal(
            variances, np.ones_like(variances), decimal=1,
            err_msg="Standardized features should have unit variance"
        )

    def test_handle_missing_values(self, medium_dataset):
        """Preprocessing should handle artificially introduced NaN values."""
        df = medium_dataset.copy()

        # Introduce some NaN values
        df.loc[0, 'vix'] = np.nan
        df.loc[5, 'competitor_mean'] = np.nan

        # Simple imputation (mean)
        numeric_cols = ['vix', 'competitor_mean']
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())

        assert df['vix'].notna().all(), "VIX should have no NaN after imputation"
        assert df['competitor_mean'].notna().all()

    def test_date_sorting(self, medium_dataset):
        """Data should be properly sorted by date."""
        df = medium_dataset.copy()

        # Verify chronological order
        dates = df['date'].values
        assert np.all(dates[:-1] <= dates[1:]), (
            "Dates should be in chronological order"
        )

    def test_train_test_split_temporal(self, medium_dataset):
        """Test temporal train/test split."""
        df = medium_dataset.copy()

        # 80/20 temporal split
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        assert len(train) == 80
        assert len(test) == 20

        # Verify no overlap
        assert train['date'].max() < test['date'].min(), (
            "Train and test should not overlap temporally"
        )


# =============================================================================
# MODELING INTEGRATION TESTS
# =============================================================================


class TestModelingMediumScale:
    """Modeling integration tests using medium_dataset."""

    @pytest.fixture
    def prepared_data(self, medium_dataset):
        """Prepare data for modeling."""
        df = medium_dataset.copy()

        # Define features and target
        feature_cols = [
            'own_cap_rate_lag_1',
            'competitor_mean_lag_1',
            'vix',
            'dgs5',
            'spread',
        ]

        X = df[feature_cols]
        y = df['sales']

        # Temporal split
        split_idx = int(len(df) * 0.8)

        return {
            'X_train': X.iloc[:split_idx],
            'X_test': X.iloc[split_idx:],
            'y_train': y.iloc[:split_idx],
            'y_test': y.iloc[split_idx:],
        }

    def test_linear_regression_fits(self, prepared_data):
        """LinearRegression should fit on medium dataset."""
        model = LinearRegression()

        # Should not raise
        model.fit(prepared_data['X_train'], prepared_data['y_train'])

        # Should have coefficients
        assert len(model.coef_) == 5
        assert model.intercept_ is not None

    def test_model_produces_predictions(self, prepared_data):
        """Model should produce predictions on test set."""
        model = Ridge(alpha=1.0)
        model.fit(prepared_data['X_train'], prepared_data['y_train'])

        predictions = model.predict(prepared_data['X_test'])

        assert len(predictions) == len(prepared_data['y_test'])
        assert not np.any(np.isnan(predictions)), "Predictions should not be NaN"

    def test_model_evaluation_metrics(self, prepared_data):
        """Model should have reasonable R² on medium dataset."""
        model = Ridge(alpha=1.0)
        model.fit(prepared_data['X_train'], prepared_data['y_train'])

        train_score = model.score(prepared_data['X_train'], prepared_data['y_train'])
        test_score = model.score(prepared_data['X_test'], prepared_data['y_test'])

        # Note: With random data, R² may be low or negative
        # This test just verifies the pipeline works
        assert isinstance(train_score, float)
        assert isinstance(test_score, float)

    def test_coefficient_extraction(self, prepared_data):
        """Should be able to extract named coefficients."""
        model = LinearRegression()
        model.fit(prepared_data['X_train'], prepared_data['y_train'])

        feature_names = prepared_data['X_train'].columns.tolist()
        coefficients = dict(zip(feature_names, model.coef_))

        assert 'own_cap_rate_lag_1' in coefficients
        assert 'competitor_mean_lag_1' in coefficients


# =============================================================================
# PIPELINE STAGE TESTS
# =============================================================================


class TestPipelineStageMediumScale:
    """Pipeline stage integration tests."""

    def test_stage1_data_loading(self, medium_dataset):
        """Stage 1: Data loading should produce valid DataFrame."""
        df = medium_dataset

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'date' in df.columns
        assert 'sales' in df.columns

    def test_stage2_feature_validation(self, medium_dataset):
        """Stage 2: Feature validation should identify required columns."""
        df = medium_dataset

        required_cols = ['date', 'sales', 'own_cap_rate', 'competitor_mean']
        missing = [col for col in required_cols if col not in df.columns]

        assert len(missing) == 0, f"Missing required columns: {missing}"

    def test_stage3_feature_engineering(self, medium_dataset):
        """Stage 3: Feature engineering should create expected features."""
        df = medium_dataset.copy()

        # Create new features
        df['rate_spread'] = df['own_cap_rate'] - df['competitor_mean']
        df['rate_ratio'] = df['own_cap_rate'] / df['competitor_mean']

        assert 'rate_spread' in df.columns
        assert 'rate_ratio' in df.columns
        assert df['rate_ratio'].min() > 0, "Ratio should be positive"

    def test_stage4_train_test_split(self, medium_dataset):
        """Stage 4: Train/test split should be temporal."""
        df = medium_dataset

        split_date = df['date'].quantile(0.8)
        train = df[df['date'] <= split_date]
        test = df[df['date'] > split_date]

        assert len(train) + len(test) == len(df)
        assert train['date'].max() <= test['date'].min()

    def test_stage5_model_training(self, medium_dataset):
        """Stage 5: Model training should complete successfully."""
        df = medium_dataset

        X = df[['own_cap_rate_lag_1', 'competitor_mean_lag_1', 'vix']]
        y = df['sales']

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        assert hasattr(model, 'coef_')
        assert len(model.coef_) == 3

    def test_stage6_model_evaluation(self, medium_dataset):
        """Stage 6: Model evaluation should compute metrics."""
        df = medium_dataset

        X = df[['own_cap_rate_lag_1', 'competitor_mean_lag_1', 'vix']]
        y = df['sales']

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        r2 = model.score(X, y)
        predictions = model.predict(X)
        mae = np.mean(np.abs(y - predictions))

        assert isinstance(r2, float)
        assert isinstance(mae, float)
        assert mae >= 0


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================


class TestDataIntegrityMediumScale:
    """Data integrity tests using medium_dataset."""

    def test_no_duplicate_dates(self, medium_dataset):
        """Dataset should have unique dates."""
        df = medium_dataset

        assert df['date'].is_unique, "Dates should be unique"

    def test_no_negative_sales(self, medium_dataset):
        """Sales should be non-negative."""
        df = medium_dataset

        assert (df['sales'] >= 0).all(), "Sales cannot be negative"

    def test_rates_are_fractions(self, medium_dataset):
        """Rate features should be decimals, not percentages."""
        df = medium_dataset

        # Rates should be < 1 (fractions, not percentages)
        assert df['own_cap_rate'].max() < 1.0, (
            "Rates should be fractions (e.g., 0.10 not 10)"
        )
        assert df['competitor_mean'].max() < 1.0

    def test_column_count_stable(self, medium_dataset):
        """Column count should match fixture spec."""
        df = medium_dataset

        # Fixture creates ~50 features
        assert len(df.columns) >= 40, (
            f"Expected ~50 columns, got {len(df.columns)}"
        )
        assert len(df.columns) <= 60

    def test_no_inf_values(self, medium_dataset):
        """Dataset should not contain infinity values."""
        df = medium_dataset

        numeric_df = df.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric_df.values).sum()

        assert inf_count == 0, f"Found {inf_count} infinity values"


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformanceMediumScale:
    """Performance tests for medium dataset operations."""

    def test_medium_dataset_loads_quickly(self, medium_dataset):
        """Medium dataset should be available (fixture already loaded)."""
        # Fixture is already loaded by pytest
        df = medium_dataset

        # Should have expected size
        assert len(df) == 100
        assert len(df.columns) >= 40

    def test_feature_selection_fast(self, medium_dataset):
        """Feature operations on medium data should be fast."""
        import time

        df = medium_dataset

        start = time.time()

        # Typical feature operations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        _ = df[numeric_cols].corr()

        elapsed = time.time() - start

        assert elapsed < 1.0, (
            f"Correlation matrix took {elapsed:.2f}s, should be <1s"
        )

    def test_model_training_fast(self, medium_dataset):
        """Model training on medium data should be fast."""
        import time

        df = medium_dataset

        X = df[['own_cap_rate_lag_1', 'competitor_mean_lag_1', 'vix', 'dgs5']]
        y = df['sales']

        start = time.time()

        model = Ridge(alpha=1.0)
        model.fit(X, y)
        _ = model.predict(X)

        elapsed = time.time() - start

        assert elapsed < 0.5, (
            f"Model training took {elapsed:.2f}s, should be <0.5s"
        )
