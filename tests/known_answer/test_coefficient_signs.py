"""
Known-Answer Tests: Coefficient Sign Constraints
=================================================

Validates that model coefficients have economically correct signs.

Sign Constraints (Economic Theory):
    Own rate (Prudential): POSITIVE [T1]
        - Higher yield attracts customers
        - Cap rate = yield for RILA products

    Competitor rate: NEGATIVE [T1]
        - Substitution effect
        - Higher competitor rates divert customers

    Lag-0 competitor: FORBIDDEN [T1]
        - Creates simultaneity bias
        - Violates causal identification

    Sales persistence: POSITIVE [T2]
        - Momentum effect from brand awareness
        - Empirically observed, not theoretical

Knowledge Tier Tags:
    [T1] = Academically validated (microeconomics, SEC filings)
    [T2] = Empirical finding from production models

References:
    - src/products/rila_methodology.py - Constraint rule definitions
    - src/validation/coefficient_patterns.py - Regex validation
    - LEAKAGE_CHECKLIST.md - Section 6: Coefficient Sign Check
"""

import re

import pytest

# =============================================================================
# SIGN CONSTRAINT DEFINITIONS [T1]
# =============================================================================

# Feature patterns and expected signs
# Format: (pattern, expected_sign, constraint_type, business_rationale)
# NOTE: Order matters - more specific patterns (forbidden) should come BEFORE
#       general patterns (negative) to ensure proper matching
SIGN_CONSTRAINTS: list[tuple[str, str, str, str]] = [
    # Lag-0 competitors: FORBIDDEN (check FIRST - before general competitor patterns)
    (
        r"^competitor.*_t0$",
        "forbidden",
        "NO_LAG_ZERO_T0",
        "Lag-0 competitor creates simultaneity bias",
    ),
    (
        r"^competitor.*_current$",
        "forbidden",
        "NO_LAG_ZERO_CURRENT",
        "Current-period competitor violates causal identification",
    ),
    (
        r"^competitor.*_lag_0$",
        "forbidden",
        "NO_LAG_ZERO_LAG0",
        "Explicit lag-0 competitor forbidden",
    ),
    # Own-rate features: POSITIVE
    (
        r"^prudential_rate",
        "positive",
        "OWN_RATE_POSITIVE",
        "Higher own cap rates attract customers (yield economics)",
    ),
    (
        r"^P_.*rate",
        "positive",
        "OWN_RATE_POSITIVE_P",
        "P_ prefix indicates Prudential rates (same logic)",
    ),
    (
        r"^own_rate",
        "positive",
        "OWN_RATE_POSITIVE_GENERIC",
        "Generic own rate feature (positive attractiveness)",
    ),
    # Competitor features: NEGATIVE (after forbidden patterns)
    (
        r"^competitor_",
        "negative",
        "COMPETITOR_NEGATIVE",
        "Higher competitor rates divert customers (substitution)",
    ),
    (
        r"^C_.*rate",
        "negative",
        "COMPETITOR_NEGATIVE_C",
        "C_ prefix indicates competitor rates (substitution effect)",
    ),
    (
        r"^c_weighted",
        "negative",
        "COMPETITOR_NEGATIVE_WEIGHTED",
        "Weighted competitor aggregate (negative substitution)",
    ),
]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_coefficient_sign(feature: str, coefficient: float) -> tuple[bool, str, str | None]:
    """
    Validate coefficient sign against constraints.

    Parameters
    ----------
    feature : str
        Feature name to validate
    coefficient : float
        Coefficient value from model

    Returns
    -------
    Tuple[bool, str, Optional[str]]
        (is_valid, reason, constraint_type)
    """
    feature_lower = feature.lower()

    for pattern, expected, constraint_type, rationale in SIGN_CONSTRAINTS:
        if re.search(pattern, feature_lower, re.IGNORECASE):
            if expected == "forbidden":
                return (
                    False,
                    f"FORBIDDEN feature pattern: {feature}. {rationale}",
                    constraint_type,
                )
            elif expected == "positive" and coefficient <= 0:
                return (
                    False,
                    f"{feature} should be positive (got {coefficient:.4f}). {rationale}",
                    constraint_type,
                )
            elif expected == "negative" and coefficient >= 0:
                return (
                    False,
                    f"{feature} should be negative (got {coefficient:.4f}). {rationale}",
                    constraint_type,
                )
            else:
                return (True, f"Valid: {constraint_type}", constraint_type)

    # No constraint defined
    return (True, "No sign constraint defined", None)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def valid_coefficient_set() -> dict[str, float]:
    """
    Return a set of economically valid coefficients.

    All coefficients conform to sign constraints from economic theory.
    """
    return {
        "prudential_rate_current": 0.0847,
        "prudential_rate_t1": 0.0523,
        "competitor_mid_t2": -0.0312,
        "competitor_top5_t3": -0.0284,
        "C_weighted_mean_t2": -0.0198,
        "sales_target_contract_t5": 0.0156,
        "dgs5": -0.0087,  # Context-dependent
        "vix": -0.0023,  # Context-dependent
    }


@pytest.fixture
def invalid_coefficient_set() -> dict[str, float]:
    """
    Return a set of economically invalid coefficients.

    Used to validate that the constraint system catches violations.
    """
    return {
        "prudential_rate_current": -0.05,  # WRONG: should be positive
        "competitor_mid_t2": 0.03,  # WRONG: should be negative
        "competitor_mid_t0": -0.02,  # WRONG: lag-0 forbidden
        "competitor_current": -0.02,  # WRONG: current forbidden
    }


# =============================================================================
# OWN-RATE SIGN TESTS [T1]
# =============================================================================


@pytest.mark.known_answer
class TestOwnRateSigns:
    """Validate own-rate coefficient signs. [T1]"""

    @pytest.mark.parametrize(
        "feature,coefficient,expected_valid",
        [
            ("prudential_rate_current", 0.08, True),
            ("prudential_rate_current", -0.08, False),
            ("prudential_rate_t1", 0.05, True),
            ("prudential_rate_t1", -0.05, False),
            ("P_rate_lag_2", 0.04, True),
            ("P_rate_lag_2", -0.04, False),
            ("own_rate_t1", 0.06, True),
            ("own_rate_t1", -0.06, False),
        ],
    )
    def test_own_rate_sign_constraint(
        self, feature: str, coefficient: float, expected_valid: bool
    ) -> None:
        """Own rate features must have positive coefficients. [T1]

        Economic rationale: Higher own cap rates attract customers.
        Cap rate = yield, so higher yield = more attractive product.
        """
        is_valid, reason, constraint_type = validate_coefficient_sign(feature, coefficient)

        assert is_valid == expected_valid, (
            f"Feature {feature} with coefficient {coefficient}: "
            f"Expected valid={expected_valid}, got valid={is_valid}. "
            f"Reason: {reason}"
        )

    def test_prudential_prefix_catches_variants(self) -> None:
        """Prudential prefix pattern catches all naming variants. [T1]"""
        variants = [
            "prudential_rate",
            "prudential_rate_current",
            "prudential_rate_t1",
            "prudential_rate_lag_0",
        ]

        for feature in variants:
            is_valid, reason, constraint_type = validate_coefficient_sign(feature, 0.05)
            assert is_valid, f"Failed to recognize {feature} as own rate"
            assert constraint_type.startswith(
                "OWN_RATE"
            ), f"Wrong constraint type {constraint_type} for {feature}"


# =============================================================================
# COMPETITOR SIGN TESTS [T1]
# =============================================================================


@pytest.mark.known_answer
class TestCompetitorSigns:
    """Validate competitor rate coefficient signs. [T1]"""

    @pytest.mark.parametrize(
        "feature,coefficient,expected_valid",
        [
            ("competitor_mid_t2", -0.03, True),
            ("competitor_mid_t2", 0.03, False),
            ("competitor_top5_t3", -0.02, True),
            ("competitor_top5_t3", 0.02, False),
            ("competitor_weighted_t2", -0.04, True),
            ("competitor_weighted_t2", 0.04, False),
            ("C_weighted_mean_t2", -0.02, True),
            ("C_weighted_mean_t2", 0.02, False),
        ],
    )
    def test_competitor_sign_constraint(
        self, feature: str, coefficient: float, expected_valid: bool
    ) -> None:
        """Competitor rate features must have negative coefficients. [T1]

        Economic rationale: Higher competitor rates divert customers.
        Standard cross-price elasticity from microeconomics.
        """
        is_valid, reason, constraint_type = validate_coefficient_sign(feature, coefficient)

        assert is_valid == expected_valid, (
            f"Feature {feature} with coefficient {coefficient}: "
            f"Expected valid={expected_valid}, got valid={is_valid}. "
            f"Reason: {reason}"
        )

    def test_competitor_prefix_catches_variants(self) -> None:
        """Competitor prefix pattern catches all naming variants. [T1]"""
        variants = [
            "competitor_mid",
            "competitor_top5",
            "competitor_weighted",
            "competitor_core_t2",
        ]

        for feature in variants:
            is_valid, reason, constraint_type = validate_coefficient_sign(feature, -0.05)
            assert is_valid, f"Failed to recognize {feature} as competitor rate"
            assert (
                "COMPETITOR" in constraint_type
            ), f"Wrong constraint type {constraint_type} for {feature}"


# =============================================================================
# LAG-0 FORBIDDEN TESTS [T1]
# =============================================================================


@pytest.mark.known_answer
class TestLag0Forbidden:
    """Validate lag-0 competitor features are forbidden. [T1]

    CRITICAL: Lag-0 competitor rates cause simultaneity bias.
    At time t, both you and competitor move simultaneously.
    This creates spurious correlation that inflates model skill.
    """

    @pytest.mark.parametrize(
        "feature",
        [
            "competitor_mid_t0",
            "competitor_top5_t0",
            "competitor_weighted_t0",
            "competitor_current",
            "competitor_mid_current",
            "competitor_lag_0",
        ],
    )
    def test_lag0_competitor_forbidden(self, feature: str) -> None:
        """Lag-0 competitor features must be rejected. [T1]

        Reference: LEAKAGE_CHECKLIST.md Section 3
        """
        is_valid, reason, constraint_type = validate_coefficient_sign(
            feature, -0.03  # Sign is correct, but lag is wrong
        )

        assert not is_valid, (
            f"Feature {feature} should be FORBIDDEN but was accepted. "
            f"Lag-0 competitor rates violate causal identification."
        )
        assert "forbidden" in reason.lower(), f"Expected 'forbidden' in: {reason}"

    def test_lagged_competitor_allowed(self) -> None:
        """Lagged competitor features (t2+) are allowed. [T1]"""
        allowed_features = [
            "competitor_mid_t2",
            "competitor_mid_t3",
            "competitor_top5_t4",
            "competitor_weighted_t5",
        ]

        for feature in allowed_features:
            is_valid, reason, constraint_type = validate_coefficient_sign(feature, -0.03)
            assert is_valid, f"Feature {feature} should be allowed but was rejected: {reason}"


# =============================================================================
# BATCH VALIDATION TESTS [T2]
# =============================================================================


@pytest.mark.known_answer
class TestBatchValidation:
    """Validate batches of coefficients. [T2]"""

    def test_valid_coefficient_set_passes(self, valid_coefficient_set: dict[str, float]) -> None:
        """All valid coefficients should pass constraints. [T2]"""
        violations = []

        for feature, coef in valid_coefficient_set.items():
            is_valid, reason, constraint_type = validate_coefficient_sign(feature, coef)
            if not is_valid:
                violations.append(f"{feature}: {reason}")

        assert (
            len(violations) == 0
        ), f"Valid coefficient set had {len(violations)} violations:\n" + "\n".join(violations)

    def test_invalid_coefficient_set_fails(self, invalid_coefficient_set: dict[str, float]) -> None:
        """All invalid coefficients should fail constraints. [T2]"""
        expected_failures = len(invalid_coefficient_set)
        actual_failures = 0

        for feature, coef in invalid_coefficient_set.items():
            is_valid, reason, constraint_type = validate_coefficient_sign(feature, coef)
            if not is_valid:
                actual_failures += 1

        assert actual_failures == expected_failures, (
            f"Expected {expected_failures} failures, got {actual_failures}. "
            f"Some invalid coefficients were not caught."
        )

    def test_constraint_coverage(self) -> None:
        """All constraint types should be testable. [T2]"""
        # Ensure we have constraints for core feature types
        core_patterns = [
            r"prudential_rate",
            r"competitor_",
            r"competitor.*_t0",
        ]

        for pattern in core_patterns:
            matching = [
                (p, e, c, r) for p, e, c, r in SIGN_CONSTRAINTS if pattern in p or pattern == p
            ]
            assert len(matching) > 0, f"No constraint defined for pattern: {pattern}"


# =============================================================================
# REGRESSION DETECTION TESTS [T2]
# =============================================================================


@pytest.mark.known_answer
class TestRegressionDetection:
    """Detect regressions in constraint enforcement. [T2]"""

    def test_constraint_count_stable(self) -> None:
        """Number of constraints should remain stable. [T2]

        If constraints are added/removed, this test fails to alert
        developers to review the change.
        """
        expected_count = 9  # Current count as of 2026-01-31

        assert len(SIGN_CONSTRAINTS) >= expected_count, (
            f"Constraint count decreased: expected at least {expected_count}, "
            f"got {len(SIGN_CONSTRAINTS)}. "
            f"Review if intentional and update expected count."
        )

    def test_all_constraints_have_rationale(self) -> None:
        """All constraints must have business rationale. [T2]"""
        for _pattern, _expected, constraint_type, rationale in SIGN_CONSTRAINTS:
            assert len(rationale) > 10, f"Constraint {constraint_type} missing meaningful rationale"

    def test_forbidden_constraints_exist(self) -> None:
        """Forbidden constraint type must exist for leakage prevention. [T1]"""
        forbidden_constraints = [
            (p, e, c, r) for p, e, c, r in SIGN_CONSTRAINTS if e == "forbidden"
        ]

        assert len(forbidden_constraints) >= 3, (
            f"Expected at least 3 forbidden constraints for lag-0 patterns, "
            f"found {len(forbidden_constraints)}"
        )
