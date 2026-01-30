"""
Unit tests for coefficient_patterns.py

Tests coefficient sign validation against economic constraints.
"""

import pytest
from src.validation.coefficient_patterns import (
    COEFFICIENT_PATTERNS,
    validate_coefficient_sign,
    validate_all_coefficients,
    get_expected_sign,
)


class TestCoefficientPatternsRegistry:
    """Tests for COEFFICIENT_PATTERNS registry."""

    def test_own_rate_pattern_defined(self):
        """Own rate pattern should be defined."""
        assert "own_rate" in COEFFICIENT_PATTERNS
        regex, expected = COEFFICIENT_PATTERNS["own_rate"]
        assert expected == "positive"

    def test_competitor_patterns_defined(self):
        """Competitor patterns should all expect negative."""
        for key in ["competitor_weighted", "competitor_core", "competitor_top", "competitor_lag"]:
            assert key in COEFFICIENT_PATTERNS
            _, expected = COEFFICIENT_PATTERNS[key]
            assert expected == "negative"

    def test_context_dependent_patterns(self):
        """Economic indicators should be context-dependent."""
        for key in ["treasury", "vix", "spread"]:
            assert key in COEFFICIENT_PATTERNS
            _, expected = COEFFICIENT_PATTERNS[key]
            assert expected == "context_dependent"

    def test_all_patterns_have_valid_structure(self):
        """All patterns should have (regex, expected_sign) tuple."""
        for name, (regex, expected) in COEFFICIENT_PATTERNS.items():
            assert isinstance(regex, str), f"{name} regex should be string"
            assert expected in ("positive", "negative", "context_dependent")


class TestValidateCoefficientSign:
    """Tests for validate_coefficient_sign function."""

    # -------------------------------------------------------------------------
    # Own rate validation (should be positive)
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("feature,coefficient", [
        ("prudential_rate_t0", 0.5),
        ("prudential_rate", 0.1),
        ("own_rate", 0.25),
        ("p_rate", 0.15),
        ("PRUDENTIAL_RATE_T1", 0.3),  # Case insensitive
    ])
    def test_own_rate_positive_valid(self, feature, coefficient):
        """Positive coefficients for own-rate features should be valid."""
        is_valid, reason = validate_coefficient_sign(feature, coefficient)
        assert is_valid, f"Expected valid for {feature}={coefficient}: {reason}"
        assert "positive" in reason.lower() or "own_rate" in reason

    @pytest.mark.parametrize("feature,coefficient", [
        ("prudential_rate_t0", -0.5),
        ("prudential_rate", 0.0),
        ("own_rate", -0.1),
    ])
    def test_own_rate_negative_invalid(self, feature, coefficient):
        """Negative/zero coefficients for own-rate features should be invalid."""
        is_valid, reason = validate_coefficient_sign(feature, coefficient)
        assert not is_valid
        assert "should be positive" in reason

    def test_own_rate_strict_mode_raises(self):
        """Strict mode should raise ValueError on violation."""
        with pytest.raises(ValueError, match="should be positive"):
            validate_coefficient_sign("prudential_rate", -0.1, strict=True)

    # -------------------------------------------------------------------------
    # Competitor validation (should be negative)
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("feature,coefficient", [
        ("competitor_weighted_t1", -0.3),
        ("competitor_mid_t2", -0.25),  # Legacy naming
        ("competitor_mean", -0.4),
        ("competitor_core", -0.15),
        ("competitor_top5", -0.2),
        ("competitor_1st", -0.35),
        ("c_weighted_mean", -0.1),
    ])
    def test_competitor_negative_valid(self, feature, coefficient):
        """Negative coefficients for competitor features should be valid."""
        is_valid, reason = validate_coefficient_sign(feature, coefficient)
        assert is_valid, f"Expected valid for {feature}={coefficient}: {reason}"

    @pytest.mark.parametrize("feature,coefficient", [
        ("competitor_weighted_t1", 0.3),
        ("competitor_mid_t2", 0.0),
        ("competitor_core", 0.15),
    ])
    def test_competitor_positive_invalid(self, feature, coefficient):
        """Positive/zero coefficients for competitor features should be invalid."""
        is_valid, reason = validate_coefficient_sign(feature, coefficient)
        assert not is_valid
        assert "should be negative" in reason

    def test_competitor_strict_mode_raises(self):
        """Strict mode should raise ValueError on violation."""
        with pytest.raises(ValueError, match="should be negative"):
            validate_coefficient_sign("competitor_weighted_t1", 0.5, strict=True)

    # -------------------------------------------------------------------------
    # Context-dependent features
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("feature,coefficient", [
        ("dgs5", 0.1),
        ("dgs5", -0.2),
        ("treasury", 0.05),
        ("econ.treasury", -0.15),
        ("vix", 0.3),
        ("volatility", -0.25),
        ("spread_t1", 0.2),
    ])
    def test_context_dependent_always_valid(self, feature, coefficient):
        """Context-dependent features should always be valid."""
        is_valid, reason = validate_coefficient_sign(feature, coefficient)
        assert is_valid
        assert "Context-dependent" in reason

    # -------------------------------------------------------------------------
    # Unknown features
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("feature,coefficient", [
        ("unknown_feature", 0.5),
        ("some_random_name", -0.3),
        ("x", 1.0),
    ])
    def test_unknown_features_no_constraint(self, feature, coefficient):
        """Unknown features should pass with 'no constraint' message."""
        is_valid, reason = validate_coefficient_sign(feature, coefficient)
        assert is_valid
        assert "No constraint" in reason

    # -------------------------------------------------------------------------
    # Product type parameter (extensibility)
    # -------------------------------------------------------------------------

    def test_product_type_parameter_accepted(self):
        """Product type parameter should be accepted (future extensibility)."""
        is_valid, _ = validate_coefficient_sign("prudential_rate", 0.5, product_type="FIA")
        assert is_valid

        is_valid, _ = validate_coefficient_sign("prudential_rate", 0.5, product_type="MYGA")
        assert is_valid


class TestValidateAllCoefficients:
    """Tests for validate_all_coefficients function."""

    def test_all_passing_coefficients(self):
        """All valid coefficients should pass."""
        coefficients = {
            "prudential_rate_t0": 0.5,
            "competitor_weighted_t1": -0.3,
            "competitor_core": -0.2,
        }
        results = validate_all_coefficients(coefficients)

        assert len(results["passed"]) == 3
        assert len(results["violated"]) == 0
        assert len(results["warnings"]) == 0

    def test_some_violations(self):
        """Violations should be captured in 'violated' list."""
        coefficients = {
            "prudential_rate": 0.5,        # Valid
            "competitor_weighted": 0.3,     # Invalid (should be negative)
            "dgs5": 0.1,                    # Context-dependent → warning
        }
        results = validate_all_coefficients(coefficients)

        assert len(results["passed"]) == 1
        assert len(results["violated"]) == 1
        assert len(results["warnings"]) == 1

        # Check violated entry
        violated = results["violated"][0]
        assert violated["feature"] == "competitor_weighted"
        assert violated["coefficient"] == 0.3
        assert "should be negative" in violated["reason"]

    def test_empty_coefficients(self):
        """Empty coefficient dict should return empty results."""
        results = validate_all_coefficients({})

        assert results["passed"] == []
        assert results["violated"] == []
        assert results["warnings"] == []

    def test_result_structure(self):
        """Results should have correct structure."""
        coefficients = {"prudential_rate": 0.5}
        results = validate_all_coefficients(coefficients)

        assert "passed" in results
        assert "violated" in results
        assert "warnings" in results

        entry = results["passed"][0]
        assert "feature" in entry
        assert "coefficient" in entry
        assert "reason" in entry


class TestGetExpectedSign:
    """Tests for get_expected_sign function."""

    @pytest.mark.parametrize("feature,expected_sign", [
        ("prudential_rate", "positive"),
        ("prudential_rate_t0", "positive"),
        ("own_rate", "positive"),
        ("competitor_weighted_t1", "negative"),
        ("competitor_mid", "negative"),
        ("competitor_core", "negative"),
        ("dgs5", "context_dependent"),
        ("vix", "context_dependent"),
        ("spread", "context_dependent"),
    ])
    def test_known_features(self, feature, expected_sign):
        """Known features should return expected sign."""
        result = get_expected_sign(feature)
        assert result == expected_sign

    def test_unknown_feature_returns_none(self):
        """Unknown features should return None."""
        assert get_expected_sign("unknown_feature") is None
        assert get_expected_sign("random_name") is None

    def test_case_insensitive(self):
        """Feature matching should be case insensitive."""
        assert get_expected_sign("PRUDENTIAL_RATE") == "positive"
        assert get_expected_sign("Competitor_Weighted") == "negative"


class TestUnifiedNaming:
    """Tests for unified feature naming (2026-01-26 refactor)."""

    def test_t0_suffix_works(self):
        """_t0 suffix (unified naming) should work."""
        is_valid, reason = validate_coefficient_sign("prudential_rate_t0", 0.5)
        assert is_valid
        assert "positive" in reason.lower()

    def test_competitor_weighted_works(self):
        """competitor_weighted (unified naming) should work."""
        is_valid, reason = validate_coefficient_sign("competitor_weighted_t2", -0.3)
        assert is_valid
        assert "negative" in reason.lower()

    def test_legacy_competitor_mid_still_works(self):
        """Legacy competitor_mid naming should still work."""
        is_valid, reason = validate_coefficient_sign("competitor_mid_t1", -0.25)
        assert is_valid

    def test_competitor_lag_pattern_covers_all_lags(self):
        """Competitor lag pattern should cover t0, t1, t2, etc."""
        for lag in range(0, 10):
            is_valid, _ = validate_coefficient_sign(f"competitor_weighted_t{lag}", -0.1)
            assert is_valid
