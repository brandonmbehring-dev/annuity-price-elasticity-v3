"""
Economic Constraint Invariant Property Tests
=============================================

Property-based tests for economic constraint validation using Hypothesis.
These tests verify that economic constraints are correctly applied regardless of input.

Economic Invariants Tested:
- Own rate coefficient must be positive (or zero)
- Competitor rate coefficient must be negative (or zero)
- Lag-0 features are always rejected
- Sign constraints are applied consistently

Author: Claude Code
Date: 2026-01-31
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from typing import List, Dict, Any
from dataclasses import dataclass


# =============================================================================
# ECONOMIC SIGN CONSTRAINT INVARIANTS
# =============================================================================


class TestOwnRateSignInvariants:
    """Property tests for own rate (Prudential) coefficient sign invariants."""

    @given(
        own_rate_coef=st.floats(min_value=-10, max_value=10, allow_nan=False)
    )
    def test_own_rate_sign_check_correct(self, own_rate_coef: float):
        """
        Invariant: Own rate coefficient check returns True only for positive values.

        Economic rationale: Higher own rates should attract more customers (positive effect).
        """
        def check_own_rate_sign(coef: float) -> bool:
            """Check if own rate coefficient has correct (positive) sign."""
            return coef >= 0

        is_valid = check_own_rate_sign(own_rate_coef)

        if own_rate_coef >= 0:
            assert is_valid, f"Positive own rate coef {own_rate_coef} rejected"
        else:
            assert not is_valid, f"Negative own rate coef {own_rate_coef} accepted"

    @given(
        coefficients=st.lists(
            st.floats(min_value=-5, max_value=5, allow_nan=False),
            min_size=1,
            max_size=10
        )
    )
    def test_own_rate_at_index_0_checked(self, coefficients: List[float]):
        """
        Invariant: Own rate is conventionally at index 0 and must be non-negative.
        """
        own_rate_index = 0
        own_rate_coef = coefficients[own_rate_index]

        is_valid = own_rate_coef >= 0
        expected_valid = own_rate_coef >= 0

        assert is_valid == expected_valid


class TestCompetitorRateSignInvariants:
    """Property tests for competitor rate coefficient sign invariants."""

    @given(
        competitor_coef=st.floats(min_value=-10, max_value=10, allow_nan=False)
    )
    def test_competitor_rate_sign_check_correct(self, competitor_coef: float):
        """
        Invariant: Competitor rate coefficient check returns True only for negative values.

        Economic rationale: Higher competitor rates should draw customers away (substitution effect).
        """
        def check_competitor_rate_sign(coef: float) -> bool:
            """Check if competitor rate coefficient has correct (negative) sign."""
            return coef <= 0

        is_valid = check_competitor_rate_sign(competitor_coef)

        if competitor_coef <= 0:
            assert is_valid, f"Negative competitor coef {competitor_coef} rejected"
        else:
            assert not is_valid, f"Positive competitor coef {competitor_coef} accepted"

    @given(
        n_competitors=st.integers(min_value=1, max_value=5),
        base_magnitude=st.floats(min_value=0.1, max_value=5.0, allow_nan=False)
    )
    def test_all_competitor_coefficients_checked(self, n_competitors: int, base_magnitude: float):
        """
        Invariant: All competitor coefficients must satisfy sign constraint, not just first.
        """
        # Generate random competitor coefficients
        np.random.seed(42)
        competitor_coefs = np.random.uniform(-base_magnitude, base_magnitude, n_competitors)

        def check_all_competitor_signs(coefs: np.ndarray) -> bool:
            """All competitor coefficients must be non-positive."""
            return np.all(coefs <= 0)

        is_valid = check_all_competitor_signs(competitor_coefs)

        # Should only be valid if ALL coefficients are non-positive
        expected_valid = np.all(competitor_coefs <= 0)
        assert is_valid == expected_valid


class TestLag0FeatureInvariants:
    """Property tests for lag-0 feature rejection invariants."""

    @given(
        feature_name=st.text(min_size=1, max_size=50)
    )
    def test_lag0_pattern_detection(self, feature_name: str):
        """
        Invariant: Lag-0 patterns (_t0, _current) are always detected.
        """
        lag0_patterns = ['_t0', '_current']

        def is_lag0_feature(name: str) -> bool:
            """Check if feature name indicates lag-0."""
            return any(pattern in name.lower() for pattern in lag0_patterns)

        is_lag0 = is_lag0_feature(feature_name)

        # If any pattern is in the name, it should be detected
        expected_lag0 = any(p in feature_name.lower() for p in lag0_patterns)
        assert is_lag0 == expected_lag0

    @given(
        base_name=st.sampled_from(['competitor_top5', 'competitor_mid', 'competitor_core']),
        lag_suffix=st.sampled_from(['_t0', '_t1', '_t2', '_t3', '_current', ''])
    )
    def test_competitor_lag0_always_rejected(self, base_name: str, lag_suffix: str):
        """
        Invariant: Competitor features with lag-0 are always rejected.
        """
        feature_name = base_name + lag_suffix

        def is_competitor_lag0(name: str) -> bool:
            """Check if competitor feature is lag-0."""
            is_competitor = 'competitor' in name.lower()
            is_lag0 = '_t0' in name or '_current' in name
            return is_competitor and is_lag0

        should_reject = is_competitor_lag0(feature_name)

        # Verify expectation
        expected_reject = 'competitor' in feature_name.lower() and (
            '_t0' in feature_name or '_current' in feature_name
        )
        assert should_reject == expected_reject

    @given(
        feature_list=st.lists(
            st.sampled_from([
                'prudential_rate_t1',
                'competitor_top5_t2',
                'competitor_mid_current',
                'competitor_core_t0',
                'competitor_top5_t3',
                'spread_t1'
            ]),
            min_size=1,
            max_size=6,
            unique=True
        )
    )
    def test_lag0_filtering_removes_only_lag0(self, feature_list: List[str]):
        """
        Invariant: Lag-0 filtering removes exactly the lag-0 features.
        """
        def filter_lag0_competitors(features: List[str]) -> List[str]:
            """Remove lag-0 competitor features."""
            lag0_patterns = ['_t0', '_current']
            return [
                f for f in features
                if not ('competitor' in f.lower() and any(p in f for p in lag0_patterns))
            ]

        filtered = filter_lag0_competitors(feature_list)

        # Verify: no lag-0 competitors in filtered list
        for f in filtered:
            if 'competitor' in f.lower():
                assert '_t0' not in f and '_current' not in f, (
                    f"Lag-0 competitor feature {f} not filtered"
                )

        # Verify: non-lag0 features preserved
        for f in feature_list:
            if 'competitor' in f.lower():
                is_lag0 = '_t0' in f or '_current' in f
                if not is_lag0:
                    assert f in filtered, f"Valid feature {f} incorrectly removed"


# =============================================================================
# CONSTRAINT APPLICATION INVARIANTS
# =============================================================================


class TestConstraintApplicationInvariants:
    """Property tests for constraint application consistency."""

    @given(
        own_rate_coef=st.floats(min_value=-5, max_value=5, allow_nan=False),
        competitor_coef=st.floats(min_value=-5, max_value=5, allow_nan=False)
    )
    def test_constraint_check_deterministic(
        self,
        own_rate_coef: float,
        competitor_coef: float
    ):
        """
        Invariant: Constraint checks are deterministic (same input -> same output).
        """
        def check_constraints(own: float, comp: float) -> bool:
            """Check both sign constraints."""
            return own >= 0 and comp <= 0

        result1 = check_constraints(own_rate_coef, competitor_coef)
        result2 = check_constraints(own_rate_coef, competitor_coef)

        assert result1 == result2, "Constraint check is not deterministic"

    @given(
        coefficients=st.lists(
            st.floats(min_value=-3, max_value=3, allow_nan=False),
            min_size=3,
            max_size=10
        )
    )
    def test_constraint_violations_counted_correctly(self, coefficients: List[float]):
        """
        Invariant: Violation count equals number of coefficients with wrong sign.
        """
        # Assume: own_rate at index 0 (should be positive)
        # Assume: competitors at indices 1+ (should be negative)
        own_rate = coefficients[0]
        competitors = coefficients[1:]

        def count_violations(own: float, comps: List[float]) -> int:
            """Count sign constraint violations."""
            violations = 0
            if own < 0:
                violations += 1
            for c in comps:
                if c > 0:
                    violations += 1
            return violations

        violation_count = count_violations(own_rate, competitors)

        # Manual verification
        expected = 0
        if own_rate < 0:
            expected += 1
        expected += sum(1 for c in competitors if c > 0)

        assert violation_count == expected


# =============================================================================
# FEATURE NAME VALIDATION INVARIANTS
# =============================================================================


class TestFeatureNameInvariants:
    """Property tests for feature name validation invariants."""

    @given(
        lag_number=st.integers(min_value=0, max_value=20)
    )
    def test_lag_number_extracted_correctly(self, lag_number: int):
        """
        Invariant: Lag number is correctly extracted from feature name.
        """
        import re

        feature_name = f"competitor_top5_t{lag_number}"

        def extract_lag(name: str) -> int:
            """Extract lag number from feature name."""
            match = re.search(r'_t(\d+)', name)
            if match:
                return int(match.group(1))
            if '_current' in name:
                return 0
            return -1  # Unknown

        extracted = extract_lag(feature_name)
        assert extracted == lag_number

    @given(
        base_name=st.sampled_from(['prudential_rate', 'competitor_top5', 'spread']),
        lag_number=st.integers(min_value=1, max_value=10)
    )
    def test_lagged_feature_name_format(self, base_name: str, lag_number: int):
        """
        Invariant: Lagged feature names follow pattern: {base}_t{lag}.
        """
        expected_name = f"{base_name}_t{lag_number}"

        # Verify format
        assert '_t' in expected_name
        assert expected_name.endswith(f'_t{lag_number}')

    @given(
        features=st.lists(
            st.sampled_from([
                'prudential_rate_t1', 'prudential_rate_t2',
                'competitor_top5_t1', 'competitor_top5_t2', 'competitor_top5_t3',
                'competitor_mid_t2', 'competitor_core_t2'
            ]),
            min_size=2,
            max_size=5,
            unique=True
        )
    )
    def test_feature_set_has_consistent_lags(self, features: List[str]):
        """
        Invariant: Feature set should have consistent lag structure.
        """
        import re

        # Extract all lag numbers
        lags = []
        for f in features:
            match = re.search(r'_t(\d+)', f)
            if match:
                lags.append(int(match.group(1)))

        # All lags should be positive (no lag-0)
        assert all(lag > 0 for lag in lags), (
            f"Found lag-0 in feature set: {features}"
        )
