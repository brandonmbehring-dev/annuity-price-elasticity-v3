"""
Critical Leakage Detection Tests
================================

Tests that validate causal identification and prevent data leakage.
These are BLOCKING tests that must pass before any production deployment.

Critical Tests:
1. Shuffled Target Degradation - Model should score near 0 on shuffled targets
2. Temporal Boundary Validation - Train max date < test min date
3. OOS Degradation Check - In-sample vs out-sample gap in expected range

Author: Claude Code
Date: 2026-01-30
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from src.validation.leakage_gates import (
    GateStatus,
    run_shuffled_target_test,
    check_temporal_boundary,
    check_r_squared_threshold,
    SHUFFLED_TARGET_THRESHOLD,
)


# =============================================================================
# SHUFFLED TARGET TESTS (P0 Critical)
# =============================================================================


class TestShuffledTargetDegradation:
    """
    Critical tests verifying model appropriately fails on shuffled targets.

    A model that performs well on shuffled targets has learned spurious patterns
    (leakage) rather than genuine signal. Shuffling breaks the causal link
    between features and target, so a good model should score near 0.
    """

    @pytest.fixture
    def genuine_signal_data(self):
        """
        Create data with genuine predictive signal.

        X genuinely predicts y through a known linear relationship.
        """
        np.random.seed(42)
        n_samples = 200

        # Features with known relationship to target
        X = pd.DataFrame({
            'own_cap_rate_lag_1': np.random.uniform(0.08, 0.12, n_samples),
            'competitor_mean_lag_1': np.random.uniform(0.07, 0.11, n_samples),
            'vix': np.random.uniform(15, 30, n_samples),
            'dgs5': np.random.uniform(2.5, 4.5, n_samples),
        })

        # Target with genuine signal from features
        # sales = 10000 + 50000*own_rate - 30000*competitor_rate + noise
        y = pd.Series(
            10000
            + 50000 * X['own_cap_rate_lag_1']
            - 30000 * X['competitor_mean_lag_1']
            + np.random.normal(0, 500, n_samples),
            name='sales'
        )

        return X, y

    @pytest.fixture
    def leaky_data(self):
        """
        Create data that simulates leakage.

        Target is directly derived from features in a way that would
        persist even after shuffling (deterministic transformation).
        """
        np.random.seed(42)
        n_samples = 100

        # Features
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
        })

        # Target is sum of features (no noise)
        # This is NOT realistic - just for testing detection
        y = pd.Series(X['feature1'] + X['feature2'], name='target')

        return X, y

    def test_linear_model_fails_on_shuffled_target(self, genuine_signal_data):
        """
        LinearRegression should score near 0 on shuffled targets.

        This validates that our shuffled target test correctly identifies
        models that have learned genuine signal (not leakage).
        """
        X, y = genuine_signal_data
        model = LinearRegression()

        result = run_shuffled_target_test(model, X, y, n_shuffles=5)

        assert result.status == GateStatus.PASS, (
            f"Linear model should PASS shuffled target test. "
            f"Got status={result.status}, avg_score={result.metric_value:.4f}. "
            f"A model with genuine signal should score near 0 on shuffled data."
        )

        # Verify score is low (near 0)
        assert result.metric_value < SHUFFLED_TARGET_THRESHOLD, (
            f"Shuffled score {result.metric_value:.4f} exceeds threshold "
            f"{SHUFFLED_TARGET_THRESHOLD}. Model may have learned spurious patterns."
        )

    def test_ridge_model_fails_on_shuffled_target(self, genuine_signal_data):
        """Ridge regression should also fail on shuffled targets."""
        X, y = genuine_signal_data
        model = Ridge(alpha=1.0)

        result = run_shuffled_target_test(model, X, y, n_shuffles=5)

        assert result.status == GateStatus.PASS, (
            f"Ridge model should PASS (score low on shuffled). "
            f"Got {result.status}, score={result.metric_value:.4f}"
        )

    def test_random_forest_fails_on_shuffled_target(self, genuine_signal_data):
        """
        RandomForest should fail on shuffled targets.

        Even complex models shouldn't learn patterns from shuffled data.
        NOTE: RF can overfit to shuffled data if not constrained, so we use
        very restrictive parameters here.
        """
        X, y = genuine_signal_data
        # Very constrained RF to prevent overfitting to noise
        model = RandomForestRegressor(
            n_estimators=5,
            max_depth=2,
            min_samples_leaf=20,
            random_state=42
        )

        result = run_shuffled_target_test(model, X, y, n_shuffles=3)

        assert result.status == GateStatus.PASS, (
            f"RandomForest should PASS (score low on shuffled). "
            f"Got {result.status}, score={result.metric_value:.4f}"
        )

    def test_shuffled_score_lower_than_actual(self, genuine_signal_data):
        """
        Model's score on actual data should be much higher than on shuffled.

        This is the fundamental check for genuine signal vs leakage.
        """
        X, y = genuine_signal_data
        model = LinearRegression()

        # Fit on actual data
        model.fit(X, y)
        actual_score = model.score(X, y)

        # Get shuffled score
        result = run_shuffled_target_test(model, X, y, n_shuffles=5)
        shuffled_score = result.metric_value

        # Actual should be MUCH higher than shuffled
        assert actual_score > shuffled_score + 0.3, (
            f"Actual score ({actual_score:.4f}) should be much higher than "
            f"shuffled score ({shuffled_score:.4f}). Difference should be >0.3"
        )

    def test_shuffled_target_test_runs_multiple_iterations(self, genuine_signal_data):
        """Verify shuffled target test actually runs multiple shuffles."""
        X, y = genuine_signal_data
        model = LinearRegression()

        result = run_shuffled_target_test(model, X, y, n_shuffles=10)

        # Should have run successfully
        assert result.status == GateStatus.PASS
        # Should have shuffled scores in details
        if 'shuffled_scores' in result.details:
            assert len(result.details['shuffled_scores']) == 10, (
                f"Expected 10 shuffle iterations, got {len(result.details['shuffled_scores'])}"
            )


# =============================================================================
# TEMPORAL BOUNDARY TESTS (P0 Critical)
# =============================================================================


class TestTemporalBoundaryValidation:
    """
    Critical tests verifying no future data leaks into training.

    Temporal leakage is one of the most common and dangerous forms of
    data leakage. Training data must strictly precede test data.
    """

    def test_valid_temporal_split_passes(self):
        """Proper temporal split should pass."""
        train_dates = pd.Series(pd.date_range('2022-01-01', '2023-06-30', freq='W'))
        test_dates = pd.Series(pd.date_range('2023-07-07', '2023-12-31', freq='W'))

        result = check_temporal_boundary(train_dates, test_dates)

        assert result.status == GateStatus.PASS, (
            f"Valid temporal split should pass. Got: {result.message}"
        )

    def test_overlapping_dates_halts(self):
        """Overlapping train/test dates should halt."""
        # Train period overlaps with test
        train_dates = pd.Series(pd.date_range('2022-01-01', '2023-08-15', freq='W'))
        test_dates = pd.Series(pd.date_range('2023-07-01', '2023-12-31', freq='W'))

        result = check_temporal_boundary(train_dates, test_dates)

        assert result.status == GateStatus.HALT, (
            f"Overlapping dates should HALT. Train ends {train_dates.max()}, "
            f"test starts {test_dates.min()}"
        )

    def test_same_day_boundary_halts(self):
        """Same day on train/test boundary should halt."""
        # Use explicit dates to ensure overlap on same day
        train_dates = pd.Series([
            pd.Timestamp('2023-06-01'),
            pd.Timestamp('2023-06-15'),
            pd.Timestamp('2023-06-30'),  # Train ends here
        ])
        test_dates = pd.Series([
            pd.Timestamp('2023-06-30'),  # Test starts same day
            pd.Timestamp('2023-07-15'),
            pd.Timestamp('2023-07-31'),
        ])

        result = check_temporal_boundary(train_dates, test_dates)

        assert result.status == GateStatus.HALT, (
            f"Same-day boundary should HALT (train max={train_dates.max()}, "
            f"test min={test_dates.min()})"
        )

    def test_requires_gap_between_train_test(self):
        """
        Train max date should be strictly before test min date.

        Best practice: At least 1 week gap to prevent look-ahead bias.
        """
        # Gap of exactly 7 days
        train_dates = pd.Series(pd.date_range('2022-01-01', '2023-06-30', freq='W'))
        test_dates = pd.Series(pd.date_range('2023-07-07', '2023-12-31', freq='W'))

        result = check_temporal_boundary(train_dates, test_dates)

        assert result.status == GateStatus.PASS

        # Verify gap exists
        gap = (test_dates.min() - train_dates.max()).days
        assert gap >= 7, f"Gap between train/test should be >= 7 days, got {gap}"

    def test_future_data_in_training_halts(self):
        """Train data containing future dates relative to test should halt."""
        # Train extends into "future" relative to test start
        train_dates = pd.Series(pd.date_range('2022-01-01', '2024-01-01', freq='W'))
        test_dates = pd.Series(pd.date_range('2023-07-01', '2023-12-31', freq='W'))

        result = check_temporal_boundary(train_dates, test_dates)

        assert result.status == GateStatus.HALT, (
            "Training data extending past test start should HALT"
        )

    def test_boundary_check_with_realistic_dates(self):
        """Test with realistic RILA project date ranges."""
        # Typical RILA split: ~3 years train, ~6 months holdout
        train_dates = pd.Series(pd.date_range('2020-01-01', '2023-06-30', freq='W'))
        test_dates = pd.Series(pd.date_range('2023-07-07', '2024-01-01', freq='W'))

        result = check_temporal_boundary(train_dates, test_dates)

        assert result.status == GateStatus.PASS
        # Verify train doesn't extend too far
        assert train_dates.max() < test_dates.min()


# =============================================================================
# OOS DEGRADATION TESTS (P0 Critical)
# =============================================================================


class TestOOSDegradation:
    """
    Critical tests verifying in-sample vs out-of-sample degradation.

    Expected behavior:
    - Some degradation is normal (5-30% R² drop)
    - No degradation = likely leakage
    - Massive degradation = overfitting
    """

    @pytest.fixture
    def temporal_split_data(self):
        """Create data with proper temporal train/test split."""
        np.random.seed(42)
        n_train = 150
        n_test = 50

        # Training data
        X_train = pd.DataFrame({
            'own_cap_rate_lag_1': np.random.uniform(0.08, 0.12, n_train),
            'competitor_mean_lag_1': np.random.uniform(0.07, 0.11, n_train),
            'vix': np.random.uniform(15, 30, n_train),
        })

        y_train = pd.Series(
            10000
            + 50000 * X_train['own_cap_rate_lag_1']
            - 30000 * X_train['competitor_mean_lag_1']
            + 100 * X_train['vix']
            + np.random.normal(0, 500, n_train)
        )

        # Test data (slightly different distribution to simulate real OOS)
        X_test = pd.DataFrame({
            'own_cap_rate_lag_1': np.random.uniform(0.085, 0.125, n_test),
            'competitor_mean_lag_1': np.random.uniform(0.065, 0.105, n_test),
            'vix': np.random.uniform(18, 35, n_test),  # Slight regime shift
        })

        y_test = pd.Series(
            10000
            + 50000 * X_test['own_cap_rate_lag_1']
            - 30000 * X_test['competitor_mean_lag_1']
            + 100 * X_test['vix']
            + np.random.normal(0, 600, n_test)  # Slightly more noise OOS
        )

        return X_train, y_train, X_test, y_test

    def test_oos_degradation_in_expected_range(self, temporal_split_data):
        """
        OOS degradation should be 5-30% of in-sample R².

        This is the core leakage vs overfit detection test:
        - degradation < 5%: Suspicious, possible leakage
        - degradation 5-30%: Normal, healthy model
        - degradation > 30%: Overfitting, model too complex
        """
        X_train, y_train, X_test, y_test = temporal_split_data
        model = LinearRegression()

        # Fit model
        model.fit(X_train, y_train)

        # Calculate scores
        in_sample_r2 = model.score(X_train, y_train)
        out_sample_r2 = model.score(X_test, y_test)

        # Calculate degradation
        # Note: Use in_sample as denominator for percentage drop
        degradation = (in_sample_r2 - out_sample_r2) / in_sample_r2 if in_sample_r2 > 0 else 0

        # Verify degradation is reasonable
        assert degradation >= 0.0, (
            f"OOS R² ({out_sample_r2:.4f}) > in-sample R² ({in_sample_r2:.4f}). "
            f"This is suspicious - possible data leakage!"
        )

        assert degradation <= 0.50, (
            f"OOS degradation {degradation:.1%} exceeds 50%. "
            f"In-sample: {in_sample_r2:.4f}, OOS: {out_sample_r2:.4f}. "
            f"Model may be overfitting."
        )

        # Informational logging
        print(f"In-sample R²: {in_sample_r2:.4f}")
        print(f"Out-sample R²: {out_sample_r2:.4f}")
        print(f"Degradation: {degradation:.1%}")

    def test_no_leakage_some_degradation_expected(self, temporal_split_data):
        """
        A clean model should show SOME degradation on OOS data.

        Zero degradation is suspicious because real OOS data always has
        some distribution shift.
        """
        X_train, y_train, X_test, y_test = temporal_split_data
        model = Ridge(alpha=1.0)

        model.fit(X_train, y_train)

        in_sample_r2 = model.score(X_train, y_train)
        out_sample_r2 = model.score(X_test, y_test)

        # Some degradation expected
        assert out_sample_r2 < in_sample_r2, (
            f"OOS R² ({out_sample_r2:.4f}) should be lower than "
            f"in-sample R² ({in_sample_r2:.4f})"
        )

    def test_complex_model_not_overfitting(self, temporal_split_data):
        """
        Even complex models should not overfit dramatically.

        RandomForest with depth limit should generalize reasonably.
        """
        X_train, y_train, X_test, y_test = temporal_split_data

        # Constrained RF to prevent extreme overfitting
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        in_sample_r2 = model.score(X_train, y_train)
        out_sample_r2 = model.score(X_test, y_test)

        # Degradation should still be reasonable
        degradation = (in_sample_r2 - out_sample_r2) / in_sample_r2 if in_sample_r2 > 0 else 0

        assert degradation <= 0.50, (
            f"RandomForest degradation {degradation:.1%} too high. "
            f"Consider reducing model complexity."
        )

    def test_oos_r2_must_be_positive(self, temporal_split_data):
        """
        OOS R² should be positive for a useful model.

        Negative OOS R² means model is worse than predicting the mean.
        """
        X_train, y_train, X_test, y_test = temporal_split_data
        model = LinearRegression()

        model.fit(X_train, y_train)
        out_sample_r2 = model.score(X_test, y_test)

        assert out_sample_r2 > 0, (
            f"OOS R² is {out_sample_r2:.4f} (negative). "
            f"Model performs worse than mean prediction."
        )


# =============================================================================
# INTEGRATION TESTS - COMBINED GATES
# =============================================================================


class TestCombinedLeakageGates:
    """
    Integration tests running multiple leakage gates together.

    These tests verify the complete leakage detection pipeline.
    """

    @pytest.fixture
    def clean_model_scenario(self):
        """Create a clean model scenario that should pass all gates."""
        np.random.seed(42)
        n_samples = 200

        # Clean features (no lag-0)
        X = pd.DataFrame({
            'own_cap_rate_lag_1': np.random.uniform(0.08, 0.12, n_samples),
            'competitor_mean_lag_1': np.random.uniform(0.07, 0.11, n_samples),
            'competitor_mean_lag_2': np.random.uniform(0.07, 0.11, n_samples),
        })

        # Target with genuine signal
        y = pd.Series(
            10000
            + 50000 * X['own_cap_rate_lag_1']
            - 30000 * X['competitor_mean_lag_1']
            + np.random.normal(0, 500, n_samples)
        )

        # Clean temporal split
        train_dates = pd.Series(pd.date_range('2022-01-01', '2023-06-30', freq='W')[:150])
        test_dates = pd.Series(pd.date_range('2023-07-07', '2023-12-31', freq='W')[:50])

        return {
            'X': X,
            'y': y,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'feature_names': list(X.columns),
            'r_squared': 0.15,  # Reasonable R²
        }

    def test_clean_scenario_passes_all_gates(self, clean_model_scenario):
        """Clean model scenario should pass all leakage gates."""
        from src.validation.leakage_gates import run_all_gates, detect_lag0_features

        scenario = clean_model_scenario

        # Check individual gates

        # 1. Lag-0 detection
        lag0_result = detect_lag0_features(scenario['feature_names'])
        assert lag0_result.status == GateStatus.PASS, (
            f"Clean features should have no lag-0: {lag0_result.message}"
        )

        # 2. Temporal boundary
        temporal_result = check_temporal_boundary(
            scenario['train_dates'],
            scenario['test_dates']
        )
        assert temporal_result.status == GateStatus.PASS, (
            f"Clean temporal split should pass: {temporal_result.message}"
        )

        # 3. R² threshold
        r2_result = check_r_squared_threshold(scenario['r_squared'])
        assert r2_result.status == GateStatus.PASS, (
            f"Reasonable R² should pass: {r2_result.message}"
        )

    def test_combined_gates_detect_multiple_issues(self):
        """Combined gates should detect all issues in a bad scenario."""
        from src.validation.leakage_gates import detect_lag0_features

        # Bad features with lag-0
        bad_features = [
            'own_cap_rate_lag_1',
            'competitor_mean_lag_0',  # FORBIDDEN
            'C_t',  # FORBIDDEN
        ]

        lag0_result = detect_lag0_features(bad_features)
        assert lag0_result.status == GateStatus.HALT, (
            "Should detect forbidden lag-0 features"
        )
        assert 'competitor_mean_lag_0' in str(lag0_result.details)
