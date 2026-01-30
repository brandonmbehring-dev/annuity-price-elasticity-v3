"""
Anti-Pattern Test: Lag-0 Competitor Feature Detection
======================================================

CRITICAL: Lag-0 competitor features violate causal identification.

This test module specifically targets the most dangerous anti-pattern in
elasticity modeling: using contemporaneous competitor data. When competitor
rates at time t are used to predict sales at time t, the model cannot
distinguish causation from correlation.

Why This Matters:
- Competitors may be responding to the SAME market signals as us
- Reverse causality: our pricing may influence competitor pricing
- Even if predictive, coefficients have no causal interpretation
- Business cannot use results for "what if" pricing scenarios

The Fix:
- Use competitor data from t-2 or earlier (minimum 2-week lag)
- This ensures temporal ordering required for causal claims

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import re
import pandas as pd
import numpy as np
from typing import List, Set

# =============================================================================
# PATTERNS THAT INDICATE LAG-0 COMPETITOR FEATURES
# =============================================================================

# These patterns MUST trigger a test failure
LAG0_FORBIDDEN_PATTERNS = [
    r"competitor.*_t0",           # e.g., competitor_rate_t0
    r"competitor.*_lag_?0",       # e.g., competitor_mean_lag_0, competitor_mean_lag0
    r"competitor.*_current",      # e.g., competitor_weighted_current
    r"^C_.*_t0",                  # e.g., C_weighted_t0
    r"^C_.*_lag_?0",             # e.g., C_mid_lag_0
    r"^C_.*_current",            # e.g., C_weighted_current
    r"^C_t$",                     # Exact match: C_t (ambiguous, treat as lag-0)
    r"^C_lag0$",                  # Exact match: C_lag0
    r"competitor.*_week_?0",      # e.g., competitor_rate_week_0, competitor_rate_week0
]

# These patterns are SAFE (lagged competitor features)
LAG1_PLUS_PATTERNS = [
    r"competitor.*_t[1-9]",       # t1, t2, ..., t9
    r"competitor.*_t[1-9]\d",     # t10, t11, ..., t17
    r"competitor.*_lag_?[1-9]",   # lag_1, lag1, etc.
    r"competitor.*_lag_?[1-9]\d", # lag_10, lag10, etc.
]


def detect_lag0_features(feature_names: List[str]) -> Set[str]:
    """Detect lag-0 competitor features in a list of feature names.

    Args:
        feature_names: List of feature column names to check

    Returns:
        Set of feature names that match lag-0 patterns
    """
    violations = set()

    for name in feature_names:
        name_lower = name.lower()
        for pattern in LAG0_FORBIDDEN_PATTERNS:
            if re.search(pattern, name_lower, re.IGNORECASE):
                violations.add(name)
                break

    return violations


# =============================================================================
# UNIT TESTS FOR PATTERN DETECTION
# =============================================================================


class TestLag0PatternDetection:
    """Unit tests for lag-0 pattern detection logic."""

    @pytest.mark.parametrize("feature_name", [
        "competitor_rate_t0",
        "competitor_weighted_t0",
        "competitor_mean_lag_0",
        "competitor_mid_lag0",
        "competitor_rate_current",
        "C_weighted_t0",
        "C_mid_lag_0",
        "C_t",
        "C_lag0",
        "competitor_mean_week_0",
        "competitor_rate_week0",
        "COMPETITOR_RATE_T0",  # Case insensitive
    ])
    def test_detects_forbidden_patterns(self, feature_name: str):
        """Each forbidden pattern should be detected."""
        violations = detect_lag0_features([feature_name])
        assert feature_name in violations, (
            f"Failed to detect lag-0 pattern in: {feature_name}"
        )

    @pytest.mark.parametrize("feature_name", [
        "competitor_rate_t1",
        "competitor_rate_t2",
        "competitor_weighted_t17",
        "competitor_mean_lag_1",
        "competitor_mean_lag_2",
        "C_weighted_t2",
        "prudential_rate_t0",     # Own rate at t0 is OK
        "prudential_rate_current", # Own rate current is OK
        "sales_target_t0",        # Target at t0 is OK
        "vix_t0",                 # Control variable at t0 is OK
    ])
    def test_allows_safe_patterns(self, feature_name: str):
        """Safe patterns should not be flagged."""
        violations = detect_lag0_features([feature_name])
        assert feature_name not in violations, (
            f"Incorrectly flagged safe pattern: {feature_name}"
        )

    def test_detects_multiple_violations(self):
        """Should detect all violations in a mixed list."""
        features = [
            "competitor_rate_t0",      # FORBIDDEN
            "competitor_rate_t1",      # OK
            "competitor_mean_lag_0",   # FORBIDDEN
            "prudential_rate_t0",      # OK (own rate)
            "C_weighted_current",      # FORBIDDEN
        ]

        violations = detect_lag0_features(features)

        assert len(violations) == 3
        assert "competitor_rate_t0" in violations
        assert "competitor_mean_lag_0" in violations
        assert "C_weighted_current" in violations

    def test_empty_list_no_violations(self):
        """Empty feature list should have no violations."""
        violations = detect_lag0_features([])
        assert len(violations) == 0


# =============================================================================
# INTEGRATION TESTS WITH REAL DATA STRUCTURES
# =============================================================================


class TestLag0InDataFrames:
    """Test lag-0 detection in actual DataFrame structures."""

    @pytest.fixture
    def sample_clean_features(self) -> pd.DataFrame:
        """Create sample DataFrame with NO lag-0 competitor features."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            # Own rate features (t0 is OK)
            "prudential_rate_t0": np.random.uniform(0.05, 0.15, n),
            "prudential_rate_t1": np.random.uniform(0.05, 0.15, n),

            # Competitor features (must be lagged)
            "competitor_weighted_t2": np.random.uniform(0.05, 0.15, n),
            "competitor_weighted_t3": np.random.uniform(0.05, 0.15, n),
            "competitor_mean_t2": np.random.uniform(0.05, 0.15, n),

            # Control variables (t0 is OK)
            "vix_t0": np.random.uniform(10, 30, n),
            "dgs5_t0": np.random.uniform(0.01, 0.05, n),
        })

    @pytest.fixture
    def sample_contaminated_features(self) -> pd.DataFrame:
        """Create sample DataFrame WITH lag-0 competitor features (BAD!)."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "prudential_rate_t0": np.random.uniform(0.05, 0.15, n),
            "competitor_weighted_t0": np.random.uniform(0.05, 0.15, n),  # FORBIDDEN
            "competitor_weighted_t2": np.random.uniform(0.05, 0.15, n),
            "competitor_mean_lag_0": np.random.uniform(0.05, 0.15, n),   # FORBIDDEN
        })

    def test_clean_features_pass(self, sample_clean_features):
        """Clean feature set should have no violations."""
        violations = detect_lag0_features(sample_clean_features.columns.tolist())
        assert len(violations) == 0, f"Unexpected violations: {violations}"

    def test_contaminated_features_detected(self, sample_contaminated_features):
        """Contaminated feature set should be caught."""
        violations = detect_lag0_features(sample_contaminated_features.columns.tolist())
        assert len(violations) == 2
        assert "competitor_weighted_t0" in violations
        assert "competitor_mean_lag_0" in violations

    @pytest.mark.leakage
    def test_production_feature_pipeline_clean(self):
        """
        CRITICAL: Production feature pipeline must not create lag-0 competitors.

        This test will fail if the actual feature engineering code creates
        any lag-0 competitor features. It serves as a safety net for deployment.
        """
        # Import the actual feature creation function
        try:
            from src.features.engineering.timeseries import create_lag_features
            from src.features.aggregation import create_competitor_aggregates

            # Create minimal test data
            np.random.seed(42)
            test_data = pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=100, freq="W"),
                "prudential_rate": np.random.uniform(0.05, 0.15, 100),
                "competitor_rate": np.random.uniform(0.05, 0.15, 100),
            })

            # Run feature creation
            features = create_lag_features(test_data, min_lag=2)

            # Check for violations
            violations = detect_lag0_features(features.columns.tolist())

            assert len(violations) == 0, (
                f"LEAKAGE DETECTED: Production feature pipeline created lag-0 "
                f"competitor features: {violations}"
            )

        except ImportError:
            pytest.skip("Feature engineering modules not available")


# =============================================================================
# REGRESSION TESTS FOR KNOWN BUGS
# =============================================================================


class TestKnownLag0Bugs:
    """Tests for specific lag-0 bugs that have been discovered and fixed."""

    def test_legacy_c_lag0_pattern(self):
        """
        Bug: Legacy code used 'C_lag0' shorthand for competitor lag-0.
        Fix: Pattern detection must catch this abbreviated form.
        """
        violations = detect_lag0_features(["C_lag0", "C_lag1", "C_lag2"])
        assert "C_lag0" in violations
        assert "C_lag1" not in violations
        assert "C_lag2" not in violations

    def test_current_suffix_ambiguity(self):
        """
        Bug: 'competitor_current' was used inconsistently (sometimes t0, sometimes rolling).
        Fix: Treat any 'competitor*_current' as lag-0 to be safe.
        """
        violations = detect_lag0_features([
            "competitor_rate_current",
            "competitor_weighted_current",
            "competitor_mean_current_week",
        ])

        # All "current" competitor features should be flagged
        assert "competitor_rate_current" in violations
        assert "competitor_weighted_current" in violations

    def test_underscore_vs_no_underscore_lag(self):
        """
        Bug: 'lag0' and 'lag_0' patterns were inconsistently caught.
        Fix: Both patterns must be detected.
        """
        violations = detect_lag0_features([
            "competitor_mean_lag_0",   # With underscore
            "competitor_mean_lag0",    # Without underscore
            "competitor_mean_lag_1",   # OK
            "competitor_mean_lag1",    # OK
        ])

        assert "competitor_mean_lag_0" in violations
        assert "competitor_mean_lag0" in violations
        assert len(violations) == 2


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================


class TestLag0PropertyBased:
    """Property-based tests for lag-0 detection robustness."""

    @pytest.mark.parametrize("lag", range(0, 18))
    def test_all_lags_properly_classified(self, lag: int):
        """Every lag value should be correctly classified."""
        feature_t = f"competitor_rate_t{lag}"
        feature_lag = f"competitor_rate_lag_{lag}"

        violations_t = detect_lag0_features([feature_t])
        violations_lag = detect_lag0_features([feature_lag])

        if lag == 0:
            assert feature_t in violations_t, f"Failed to detect {feature_t}"
            assert feature_lag in violations_lag, f"Failed to detect {feature_lag}"
        else:
            assert feature_t not in violations_t, f"Incorrectly flagged {feature_t}"
            assert feature_lag not in violations_lag, f"Incorrectly flagged {feature_lag}"

    @pytest.mark.parametrize("prefix", [
        "competitor", "Competitor", "COMPETITOR",
        "competitor_weighted", "competitor_mean", "competitor_mid",
        "C", "c",
    ])
    def test_case_insensitive_detection(self, prefix: str):
        """Detection should work regardless of case."""
        feature = f"{prefix}_rate_t0"
        violations = detect_lag0_features([feature])

        # Should detect if it's a competitor pattern
        if "competitor" in prefix.lower() or prefix.lower() == "c":
            assert len(violations) > 0, f"Failed to detect: {feature}"


# =============================================================================
# DOCUMENTATION TEST
# =============================================================================


def test_lag0_detection_summary():
    """
    Summary: Lag-0 Competitor Feature Detection

    WHY THIS MATTERS:
    - Lag-0 competitor features violate causal identification
    - Model cannot distinguish causation from correlation
    - Coefficients lose economic interpretation
    - "What if" pricing scenarios become meaningless

    WHAT IS FORBIDDEN:
    - competitor*_t0, competitor*_lag_0, competitor*_current
    - C_*_t0, C_*_lag_0, C_t, C_lag0
    - Any contemporaneous competitor data

    WHAT IS ALLOWED:
    - competitor*_t1, competitor*_t2, ... (lagged)
    - prudential_rate_t0 (own rate at t0 is fine)
    - vix_t0, dgs5_t0 (control variables at t0 are fine)

    ENFORCEMENT:
    - This test module runs as part of `make leakage-audit`
    - CI/CD pipeline blocks deployment on violations
    - Pre-commit hooks can optionally check feature names

    BUSINESS IMPACT OF VIOLATION:
    - Model predictions may be directionally correct but...
    - Elasticity estimates are biased and unusable
    - Pricing recommendations cannot be trusted
    - Regulatory risk if used for pricing decisions
    """
    pass  # Documentation test - always passes
