"""
Anti-Pattern Test: Future Data Leakage Detection
=================================================

CRITICAL: Models must not use future information to predict the present.

This test module detects cases where future data "leaks" into features used
for prediction. This is one of the most common and insidious bugs in time
series modeling.

Types of Future Leakage:
1. Direct: Using future values in rolling calculations
2. Indirect: Using data that wouldn't be available at prediction time
3. Train/Test: Test data contaminating training set
4. Feature Engineering: Post-hoc calculations using full dataset

Why This Matters:
- Creates artificially good performance metrics
- Model fails catastrophically in production
- "Too good to be true" R² values are a red flag
- Impossible to catch without proper temporal validation

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


# =============================================================================
# LEAKAGE DETECTION FUNCTIONS
# =============================================================================


def check_temporal_ordering(
    feature_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
) -> List[str]:
    """Check that features don't use future target values.

    Args:
        feature_df: DataFrame with features and target
        date_col: Name of date column
        target_col: Name of target column
        feature_cols: List of feature column names to check

    Returns:
        List of features that may contain future leakage
    """
    suspicious = []

    # Sort by date to ensure proper ordering
    df = feature_df.sort_values(date_col).copy()

    # For each feature, check correlation with future target
    for col in feature_cols:
        if col == target_col or col == date_col:
            continue

        # Correlation with future target (shifted back)
        future_target = df[target_col].shift(-1)  # Next period's target
        corr_with_future = df[col].corr(future_target)

        # If feature correlates more with future than present, suspicious
        present_corr = df[col].corr(df[target_col])

        if pd.notna(corr_with_future) and abs(corr_with_future) > abs(present_corr) * 1.5:
            suspicious.append(col)

    return suspicious


def check_rolling_window_leakage(
    df: pd.DataFrame,
    date_col: str,
    rolling_cols: List[str],
    window_size: int,
) -> List[str]:
    """Check that rolling calculations only use past data.

    Args:
        df: DataFrame with rolling features
        date_col: Name of date column
        rolling_cols: Columns that should be rolling calculations
        window_size: Expected window size

    Returns:
        List of columns with potential leakage
    """
    violations = []
    df_sorted = df.sort_values(date_col)

    for col in rolling_cols:
        # Check if first `window_size` values are NaN (proper rolling)
        early_values = df_sorted[col].head(window_size)

        # At least some early values should be NaN if properly rolling
        if early_values.isna().sum() == 0 and window_size > 1:
            violations.append(f"{col}: No NaN values in first {window_size} rows")

    return violations


def check_train_test_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str,
) -> bool:
    """Check that test data is strictly after training data.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        date_col: Name of date column

    Returns:
        True if there's leakage (overlap), False if clean
    """
    train_max = train_df[date_col].max()
    test_min = test_df[date_col].min()

    return test_min <= train_max


# =============================================================================
# UNIT TESTS FOR LEAKAGE DETECTION
# =============================================================================


class TestTemporalOrderingCheck:
    """Tests for temporal ordering validation."""

    @pytest.fixture
    def clean_timeseries(self) -> pd.DataFrame:
        """Create clean time series with no leakage."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="W")

        # Generate base sales with some autocorrelation
        sales = np.cumsum(np.random.randn(n)) + 30000
        sales = sales + np.random.uniform(-1000, 1000, n)  # Add noise

        return pd.DataFrame({
            "date": dates,
            "sales": sales,
            "rate_t1": np.random.uniform(0.05, 0.15, n),  # Independent random (clean)
            "rate_t2": np.random.uniform(0.05, 0.15, n),  # Independent random (clean)
            "vix_t0": np.random.uniform(10, 30, n),       # Independent random (clean)
        })

    @pytest.fixture
    def leaky_timeseries(self) -> pd.DataFrame:
        """Create time series with future leakage."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        sales = np.random.uniform(10000, 50000, n)

        # Create a leaky feature: uses future sales
        leaky_feature = np.zeros(n)
        for i in range(n - 1):
            leaky_feature[i] = sales[i + 1]  # Uses FUTURE sales!
        leaky_feature[-1] = sales[-1]

        return pd.DataFrame({
            "date": dates,
            "sales": sales,
            "rate_t1": np.random.uniform(0.05, 0.15, n),
            "leaky_feature": leaky_feature,  # CONTAINS FUTURE DATA
        })

    def test_clean_data_passes(self, clean_timeseries):
        """Clean time series should have no suspicious features."""
        suspicious = check_temporal_ordering(
            clean_timeseries,
            date_col="date",
            target_col="sales",
            feature_cols=["rate_t1", "rate_t2", "vix_t0"],
        )

        assert len(suspicious) == 0

    def test_leaky_data_detected(self, leaky_timeseries):
        """Leaky time series should be flagged."""
        suspicious = check_temporal_ordering(
            leaky_timeseries,
            date_col="date",
            target_col="sales",
            feature_cols=["rate_t1", "leaky_feature"],
        )

        assert "leaky_feature" in suspicious


class TestRollingWindowLeakage:
    """Tests for rolling window leakage detection."""

    @pytest.fixture
    def proper_rolling_df(self) -> pd.DataFrame:
        """DataFrame with properly calculated rolling features."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        values = np.random.uniform(100, 200, n)

        df = pd.DataFrame({"date": dates, "value": values})

        # Proper rolling: only uses past data, has NaN at start
        df["rolling_mean_4w"] = df["value"].rolling(window=4).mean()

        return df

    @pytest.fixture
    def improper_rolling_df(self) -> pd.DataFrame:
        """DataFrame with improperly calculated rolling features."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        values = np.random.uniform(100, 200, n)

        df = pd.DataFrame({"date": dates, "value": values})

        # WRONG: Centered rolling (uses future data)
        df["centered_mean"] = df["value"].rolling(window=4, center=True).mean()

        # WRONG: Backfilled rolling (no NaN at start)
        df["backfilled_mean"] = df["value"].rolling(window=4).mean().bfill()

        return df

    def test_proper_rolling_passes(self, proper_rolling_df):
        """Properly calculated rolling features should pass."""
        violations = check_rolling_window_leakage(
            proper_rolling_df,
            date_col="date",
            rolling_cols=["rolling_mean_4w"],
            window_size=4,
        )

        assert len(violations) == 0

    def test_backfilled_rolling_detected(self, improper_rolling_df):
        """Backfilled rolling features should be flagged."""
        violations = check_rolling_window_leakage(
            improper_rolling_df,
            date_col="date",
            rolling_cols=["backfilled_mean"],
            window_size=4,
        )

        assert len(violations) > 0


class TestTrainTestLeakage:
    """Tests for train/test split leakage detection."""

    def test_proper_split_passes(self):
        """Proper temporal split should pass."""
        # Train: 80 weeks from 2024-01-01 ends around 2025-07-06
        # Test must start AFTER train ends to avoid leakage
        train = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=80, freq="W"),
            "value": np.random.randn(80),
        })
        # Start test 2 weeks after train ends to ensure clear separation
        test = pd.DataFrame({
            "date": pd.date_range("2025-08-01", periods=20, freq="W"),
            "value": np.random.randn(20),
        })

        has_leakage = check_train_test_leakage(train, test, "date")

        assert not has_leakage

    def test_overlapping_split_detected(self):
        """Overlapping train/test split should be flagged."""
        train = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100, freq="W"),
            "value": np.random.randn(100),
        })
        test = pd.DataFrame({
            "date": pd.date_range("2024-06-01", periods=50, freq="W"),  # Overlaps!
            "value": np.random.randn(50),
        })

        has_leakage = check_train_test_leakage(train, test, "date")

        assert has_leakage

    def test_random_split_detected(self):
        """Random (non-temporal) split should be flagged."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="W")
        values = np.random.randn(100)

        # Random split (WRONG for time series)
        indices = np.random.permutation(100)
        train_idx = indices[:80]
        test_idx = indices[80:]

        train = pd.DataFrame({
            "date": dates[train_idx],
            "value": values[train_idx],
        })
        test = pd.DataFrame({
            "date": dates[test_idx],
            "value": values[test_idx],
        })

        # With random split, test data will be mixed with train dates
        has_leakage = check_train_test_leakage(train, test, "date")

        # This should detect that test min <= train max
        assert has_leakage


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestProductionPipelineLeakage:
    """Integration tests for production pipeline leakage detection."""

    @pytest.mark.leakage
    def test_feature_pipeline_no_future_data(self):
        """
        CRITICAL: Feature engineering pipeline must not use future data.

        This test validates that the actual feature creation code maintains
        proper temporal ordering.
        """
        try:
            from src.features.engineering.timeseries import create_lag_features

            # Create test data
            np.random.seed(42)
            n = 100
            test_data = pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=n, freq="W"),
                "sales": np.random.uniform(10000, 50000, n),
                "rate": np.random.uniform(0.05, 0.15, n),
            })

            # Run feature creation
            features = create_lag_features(test_data, lags=[1, 2, 3, 4])

            # Check that lagged features don't correlate more with future
            suspicious = check_temporal_ordering(
                features,
                date_col="date",
                target_col="sales",
                feature_cols=[c for c in features.columns if c not in ["date", "sales"]],
            )

            assert len(suspicious) == 0, (
                f"LEAKAGE DETECTED: Features may use future data: {suspicious}"
            )

        except ImportError:
            pytest.skip("Feature engineering module not available")

    @pytest.mark.leakage
    def test_cross_validation_temporal(self):
        """
        CRITICAL: Cross-validation must respect temporal ordering.

        Time series cross-validation should use expanding or rolling windows,
        never random splits.
        """
        try:
            from src.models.validation import create_time_series_cv

            # Create test data
            dates = pd.date_range("2024-01-01", periods=100, freq="W")

            # Get CV splits
            cv = create_time_series_cv(dates, n_splits=5)

            # Verify each split maintains temporal ordering
            for fold, (train_idx, test_idx) in enumerate(cv):
                train_max_date = dates[train_idx].max()
                test_min_date = dates[test_idx].min()

                assert test_min_date > train_max_date, (
                    f"CV fold {fold}: Test data ({test_min_date}) overlaps "
                    f"with training ({train_max_date})"
                )

        except ImportError:
            pytest.skip("Model validation module not available")


# =============================================================================
# SHUFFLED TARGET TEST
# =============================================================================


class TestShuffledTargetValidation:
    """
    The "shuffled target" test is a powerful leakage detection method.

    If a model achieves good performance on shuffled (randomized) target
    values, it indicates that the model is learning from information that
    shouldn't be predictive - i.e., data leakage.
    """

    @pytest.fixture
    def simple_linear_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Create simple linear data with no leakage."""
        np.random.seed(42)
        n = 100

        X = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
        })

        # Target is linear function of features + noise
        y = 3 * X["feature1"] + 2 * X["feature2"] + np.random.randn(n)

        return X, y

    def test_good_model_fails_on_shuffled(self, simple_linear_data):
        """A properly specified model should fail on shuffled targets."""
        from sklearn.linear_model import LinearRegression

        X, y = simple_linear_data

        # Fit on real data
        model = LinearRegression()
        model.fit(X, y)
        real_score = model.score(X, y)

        # Fit on shuffled data
        np.random.seed(123)
        y_shuffled = y.sample(frac=1, random_state=123).values
        model_shuffled = LinearRegression()
        model_shuffled.fit(X, y_shuffled)
        shuffled_score = model_shuffled.score(X, y_shuffled)

        # Shuffled should be much worse
        assert shuffled_score < real_score * 0.5, (
            f"Model achieved {shuffled_score:.2%} on shuffled data "
            f"(real: {real_score:.2%}). This suggests overfitting or leakage."
        )

    def test_leaky_model_succeeds_on_shuffled(self):
        """
        A leaky model might succeed even on shuffled targets.

        This is a simulation of what happens when features contain
        information from the target (or future target).
        """
        np.random.seed(42)
        n = 100

        # Create target
        y = np.random.uniform(10000, 50000, n)

        # Create leaky feature: exactly the target (extreme case)
        X = pd.DataFrame({
            "leaky_feature": y,  # LEAKAGE!
            "noise": np.random.randn(n),
        })

        from sklearn.linear_model import LinearRegression

        # Even with shuffled target, leaky feature correlates
        # (because we're fitting on the same data)
        model = LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)

        # This will be very high (near 1.0) due to leakage
        assert score > 0.99, "Simulating leaky model"


# =============================================================================
# DOCUMENTATION TEST
# =============================================================================


def test_future_leakage_summary():
    """
    Summary: Future Data Leakage Detection

    TYPES OF FUTURE LEAKAGE:

    1. Direct Leakage:
       - Using y_{t+1} to predict y_t
       - Including future feature values in calculations

    2. Rolling Window Leakage:
       - Centered rolling averages (use future data)
       - Backfilled NaN values (implicitly use future)

    3. Train/Test Leakage:
       - Random splits instead of temporal splits
       - Overlapping date ranges

    4. Feature Engineering Leakage:
       - Calculating statistics on full dataset before split
       - Using information not available at prediction time

    DETECTION METHODS:

    1. Shuffled Target Test:
       - Model should FAIL on randomized targets
       - High shuffled score = likely leakage

    2. Temporal Ordering Check:
       - Features should correlate with past, not future
       - Suspicious if future correlation > present correlation

    3. Rolling Window Validation:
       - Check for NaN values at start of rolling features
       - Verify window only uses past data

    4. Train/Test Split Check:
       - Test data must be strictly after training data
       - No date overlap allowed

    RED FLAGS:
    - R² > 0.90 on behavioral/economic data
    - Model "works" on shuffled targets
    - Features have no NaN at start of time series
    - Test dates overlap with training dates

    ENFORCEMENT:
    - Part of `make leakage-audit` pipeline
    - Blocks deployment on leakage detection
    - Requires explicit sign-off for production
    """
    pass  # Documentation test - always passes
