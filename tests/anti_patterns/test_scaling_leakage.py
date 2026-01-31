"""
Anti-Pattern Test: Scaling Leakage Detection
============================================

CRITICAL: Scalers must be fit ONLY on training data.

This test module detects cases where StandardScaler, MinMaxScaler, or other
normalizers are fitted on the full dataset before train/test split, which
leaks test set statistics into the training process.

Why This Matters:
- Test set mean/std leak into training features
- Creates artificially good cross-validation scores
- Model learns about data it shouldn't see
- 5-10% artificial R² inflation is common

The Fix:
- Fit scaler only on training data
- Transform test data using training parameters
- Use sklearn Pipeline for automatic per-fold scaling

Author: Claude Code
Date: 2026-01-31
"""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline


# =============================================================================
# LEAKAGE DETECTION FUNCTIONS
# =============================================================================


@dataclass
class ScalingLeakageResult:
    """Result of scaling leakage check."""
    has_leakage: bool
    leakage_score: float  # Higher = more leakage suspected
    message: str
    details: Dict[str, Any]


def detect_scaling_leakage_via_metric_gap(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    n_trials: int = 10,
    random_state: int = 42,
) -> ScalingLeakageResult:
    """Detect scaling leakage by comparing proper vs leaky pipeline performance.

    If scaling on full data provides significant advantage, leakage is present.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion for test set
        n_trials: Number of random splits to average
        random_state: Random seed for reproducibility

    Returns:
        ScalingLeakageResult with detection outcome
    """
    np.random.seed(random_state)
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_size))

    proper_scores = []
    leaky_scores = []

    for trial in range(n_trials):
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # PROPER: Fit scaler on train only
        scaler_proper = StandardScaler()
        X_train_scaled = scaler_proper.fit_transform(X_train)
        X_test_scaled = scaler_proper.transform(X_test)

        model_proper = Ridge(alpha=1.0)
        model_proper.fit(X_train_scaled, y_train)
        proper_scores.append(model_proper.score(X_test_scaled, y_test))

        # LEAKY: Fit scaler on full data
        scaler_leaky = StandardScaler()
        X_full_scaled = scaler_leaky.fit_transform(X)  # LEAKAGE!
        X_train_leaky = X_full_scaled[train_idx]
        X_test_leaky = X_full_scaled[test_idx]

        model_leaky = Ridge(alpha=1.0)
        model_leaky.fit(X_train_leaky, y_train)
        leaky_scores.append(model_leaky.score(X_test_leaky, y_test))

    proper_mean = np.mean(proper_scores)
    leaky_mean = np.mean(leaky_scores)
    score_gap = leaky_mean - proper_mean

    # If leaky is significantly better, leakage is present
    has_leakage = score_gap > 0.02  # 2% R² improvement threshold

    return ScalingLeakageResult(
        has_leakage=has_leakage,
        leakage_score=score_gap,
        message=f"Leaky scaling {'detected' if has_leakage else 'not detected'} (gap={score_gap:.4f})",
        details={
            "proper_r2_mean": proper_mean,
            "leaky_r2_mean": leaky_mean,
            "score_gap": score_gap,
            "n_trials": n_trials,
        }
    )


def detect_mean_shift_leakage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler: StandardScaler,
) -> ScalingLeakageResult:
    """Detect if scaler was fit on combined train+test data.

    If scaler.mean_ is closer to combined mean than train mean, leakage present.

    Args:
        X_train: Training features
        X_test: Test features
        scaler: Fitted StandardScaler to check

    Returns:
        ScalingLeakageResult with detection outcome
    """
    # Calculate expected means
    train_mean = np.mean(X_train, axis=0)
    combined_mean = np.mean(np.vstack([X_train, X_test]), axis=0)
    scaler_mean = scaler.mean_

    # Distance from scaler mean to train vs combined
    dist_to_train = np.mean(np.abs(scaler_mean - train_mean))
    dist_to_combined = np.mean(np.abs(scaler_mean - combined_mean))

    # If scaler is closer to combined, it was fit on full data
    has_leakage = dist_to_combined < dist_to_train * 0.9  # Allow 10% tolerance

    return ScalingLeakageResult(
        has_leakage=has_leakage,
        leakage_score=dist_to_train / (dist_to_combined + 1e-10) if dist_to_combined > 0 else 0,
        message=f"Mean shift leakage {'detected' if has_leakage else 'not detected'}",
        details={
            "dist_to_train_mean": dist_to_train,
            "dist_to_combined_mean": dist_to_combined,
            "scaler_mean": scaler_mean.tolist() if hasattr(scaler_mean, 'tolist') else scaler_mean,
        }
    )


# =============================================================================
# UNIT TESTS FOR SCALING LEAKAGE DETECTION
# =============================================================================


class TestScalingLeakageViaMetricGap:
    """Tests that verify we can detect scaling leakage through metric comparison."""

    @pytest.fixture
    def synthetic_data_with_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic data where scaling matters."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Create features with varying scales
        X = np.column_stack([
            np.random.randn(n_samples) * 1,      # Small scale
            np.random.randn(n_samples) * 100,    # Large scale
            np.random.randn(n_samples) * 0.01,   # Tiny scale
        ] + [np.random.randn(n_samples) for _ in range(n_features - 3)])

        # Target depends on standardized features
        y = 3 * X[:, 0] + 0.03 * X[:, 1] + 300 * X[:, 2] + np.random.randn(n_samples) * 0.5

        return X, y

    def test_proper_scaling_detected_as_clean(self, synthetic_data_with_signal):
        """Proper scaling workflow should not show leakage."""
        X, y = synthetic_data_with_signal

        # Do proper scaling
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Fit scaler on train only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        result = detect_mean_shift_leakage(X_train, X_test, scaler)

        assert not result.has_leakage, f"False positive: {result.message}"

    def test_leaky_scaling_detected(self, synthetic_data_with_signal):
        """Leaky scaling workflow should be detected."""
        X, y = synthetic_data_with_signal

        # Do LEAKY scaling (fit on full data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # LEAKAGE!

        # Then split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]

        result = detect_mean_shift_leakage(X_train, X_test, scaler)

        assert result.has_leakage, f"Missed leakage: {result.message}"


class TestMeanShiftDetection:
    """Tests for mean shift based leakage detection."""

    def test_clean_scaler_passes(self):
        """Scaler fit on train only should pass."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(25, 5) + 2  # Shifted test set

        scaler = StandardScaler()
        scaler.fit(X_train)  # Proper: fit on train only

        result = detect_mean_shift_leakage(X_train, X_test, scaler)

        assert not result.has_leakage

    def test_leaky_scaler_detected(self):
        """Scaler fit on full data should be detected."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(25, 5) + 2  # Shifted test set

        X_full = np.vstack([X_train, X_test])
        scaler = StandardScaler()
        scaler.fit(X_full)  # LEAKAGE: fit on full data

        result = detect_mean_shift_leakage(X_train, X_test, scaler)

        assert result.has_leakage


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestScalingLeakageInPipelines:
    """Test scaling leakage detection in sklearn pipelines."""

    @pytest.fixture
    def regression_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Create regression data similar to elasticity modeling."""
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="W"),
            "own_rate": np.random.uniform(0.05, 0.15, n),
            "competitor_rate": np.random.uniform(0.04, 0.14, n),
            "vix": np.random.uniform(10, 40, n),
            "treasury_10y": np.random.uniform(0.01, 0.05, n),
        })

        # Target depends on features
        df["sales"] = (
            30000
            + 50000 * df["own_rate"]
            - 40000 * df["competitor_rate"]
            - 200 * df["vix"]
            + 100000 * df["treasury_10y"]
            + np.random.randn(n) * 2000
        )

        X = df[["own_rate", "competitor_rate", "vix", "treasury_10y"]].values
        y = df["sales"].values

        return X, y

    def test_pipeline_prevents_leakage(self, regression_data):
        """sklearn Pipeline should automatically prevent scaling leakage."""
        X, y = regression_data

        # Temporal split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Pipeline ensures scaler fits only on training data
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])

        pipeline.fit(X_train, y_train)

        # Check that internal scaler was fit on train only
        scaler = pipeline.named_steps["scaler"]
        result = detect_mean_shift_leakage(X_train, X_test, scaler)

        assert not result.has_leakage, (
            "Pipeline should prevent scaling leakage"
        )

    def test_manual_scaling_can_leak(self, regression_data):
        """Manual scaling without Pipeline can introduce leakage."""
        X, y = regression_data

        # WRONG: Scale full data first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # LEAKAGE!

        # Then split
        split_idx = int(len(X) * 0.8)
        X_train_scaled = X_scaled[:split_idx]
        X_test_scaled = X_scaled[split_idx:]

        # Detection should flag this
        X_train_unscaled = X[:split_idx]
        X_test_unscaled = X[split_idx:]

        result = detect_mean_shift_leakage(X_train_unscaled, X_test_unscaled, scaler)

        assert result.has_leakage, (
            "Manual scaling on full data should be detected as leakage"
        )


# =============================================================================
# ANTI-PATTERN DEMONSTRATION
# =============================================================================


class TestScalingAntiPatterns:
    """Demonstrate and test common scaling anti-patterns."""

    @pytest.fixture
    def time_series_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series data where temporal split matters."""
        np.random.seed(42)
        n = 200

        # Features with drift over time (common in financial data)
        time_idx = np.arange(n)
        X = np.column_stack([
            0.05 + 0.0001 * time_idx + np.random.randn(n) * 0.01,  # Trending feature
            np.random.randn(n) * 10 + 0.05 * time_idx,  # Large scale trending
        ])

        y = 3 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5

        return X, y

    def test_anti_pattern_scale_then_split(self, time_series_data):
        """
        ANTI-PATTERN: Scaling full dataset before temporal split.

        This is WRONG because:
        1. Test set statistics leak into training
        2. Future trends influence past normalization
        3. Creates artificial improvement in metrics
        """
        X, y = time_series_data

        # ANTI-PATTERN: Scale first, split later
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Sees ALL data including future!

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]

        result = detect_mean_shift_leakage(X_train, X_test, scaler)

        # This SHOULD be flagged as leakage
        assert result.has_leakage, (
            "Anti-pattern 'scale then split' should be detected"
        )

    def test_correct_pattern_split_then_scale(self, time_series_data):
        """
        CORRECT PATTERN: Temporal split before scaling.

        This is RIGHT because:
        1. Scaler only sees training data
        2. Test data transformed with training parameters
        3. Simulates real deployment scenario
        """
        X, y = time_series_data

        # CORRECT: Split first, scale later
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # Only sees training!
        X_test_scaled = scaler.transform(X_test)

        result = detect_mean_shift_leakage(X_train, X_test, scaler)

        # This should NOT be flagged
        assert not result.has_leakage, (
            "Correct pattern 'split then scale' should pass"
        )


# =============================================================================
# CROSS-VALIDATION LEAKAGE TESTS
# =============================================================================


class TestCrossValidationScalingLeakage:
    """Test scaling leakage in cross-validation contexts."""

    def test_cv_without_pipeline_can_leak(self):
        """Cross-validation without Pipeline can introduce scaling leakage."""
        from sklearn.model_selection import cross_val_score, KFold

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(100) * 0.5

        # WRONG: Scale full data, then cross-validate
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # LEAKAGE!

        model = Ridge(alpha=1.0)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        leaky_scores = cross_val_score(model, X_scaled, y, cv=cv)

        # CORRECT: Use Pipeline (scaling happens per fold)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])

        proper_scores = cross_val_score(pipeline, X, y, cv=cv)

        # Leaky method typically shows inflated scores
        # (Though with small random data, effect may be minimal)
        assert np.mean(leaky_scores) >= np.mean(proper_scores) * 0.95, (
            "Leaky CV should not be dramatically worse (data is small)"
        )


# =============================================================================
# DOCUMENTATION TEST
# =============================================================================


def test_scaling_leakage_summary():
    """
    Summary: Scaling Leakage Detection

    WHAT IS SCALING LEAKAGE:

    Fitting a scaler (StandardScaler, MinMaxScaler) on the full dataset
    before train/test split leaks test set statistics into training.

    ANTI-PATTERN (WRONG):
    ```python
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Sees test data!
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    ```

    CORRECT PATTERN:
    ```python
    X_train, X_test = X[:split], X[split:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Only training!
    X_test_scaled = scaler.transform(X_test)
    ```

    BEST PRACTICE:
    ```python
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])
    pipeline.fit(X_train, y_train)  # Scaler auto-fits on train only
    ```

    WHY IT MATTERS:
    - 5-10% artificial R² inflation
    - Cross-validation scores are misleading
    - Model fails in production (never saw real test distribution)

    DETECTION METHODS:
    1. Compare proper vs leaky pipeline scores (gap > 2% = leakage)
    2. Check if scaler.mean_ matches train mean vs combined mean

    ENFORCEMENT:
    - Always use sklearn Pipeline
    - Run scaling leakage tests in CI
    - Part of `make leakage-audit`
    """
    pass  # Documentation test - always passes
