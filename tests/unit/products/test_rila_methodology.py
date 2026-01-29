"""
Unit Tests for RILA Methodology
===============================

Tests for src/products/rila_methodology.py covering:
- Constraint rule definitions
- Coefficient sign expectations
- Economic rationale validation
- Leakage pattern detection
- Regime detection support

Target: 85% coverage for rila_methodology.py

Test Pattern:
- Test constraint rules completeness
- Test pattern matching logic
- Test economic constraints
- Test protocol implementation

Author: Claude Code
Date: 2026-01-29
"""

import re
import pytest
from src.products.rila_methodology import RILAMethodology
from src.core.registry import ConstraintRule


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def rila_methodology():
    """RILA methodology instance for testing."""
    return RILAMethodology()


# =============================================================================
# BASIC PROPERTY TESTS
# =============================================================================


def test_product_type_property(rila_methodology):
    """Test that product_type returns 'rila'."""
    assert rila_methodology.product_type == "rila"


def test_supports_regime_detection(rila_methodology):
    """Test that RILA does not support regime detection."""
    assert rila_methodology.supports_regime_detection() == False


# =============================================================================
# CONSTRAINT RULES TESTS
# =============================================================================


def test_get_constraint_rules_returns_list(rila_methodology):
    """Test that get_constraint_rules returns a list."""
    rules = rila_methodology.get_constraint_rules()
    assert isinstance(rules, list)


def test_get_constraint_rules_non_empty(rila_methodology):
    """Test that constraint rules list is not empty."""
    rules = rila_methodology.get_constraint_rules()
    assert len(rules) > 0


def test_all_constraint_rules_are_constraint_rule_objects(rila_methodology):
    """Test that all rules are ConstraintRule instances."""
    rules = rila_methodology.get_constraint_rules()
    for rule in rules:
        assert isinstance(rule, ConstraintRule)


def test_constraint_rules_count(rila_methodology):
    """Test that RILA has expected number of constraint rules."""
    rules = rila_methodology.get_constraint_rules()
    # Should have 5 rules: own_rate (2 patterns) + competitor (2 patterns) + lag-0 forbidden
    assert len(rules) == 5


def test_constraint_rules_have_required_fields(rila_methodology):
    """Test that all constraint rules have required fields."""
    rules = rila_methodology.get_constraint_rules()

    for rule in rules:
        assert hasattr(rule, 'feature_pattern')
        assert hasattr(rule, 'expected_sign')
        assert hasattr(rule, 'constraint_type')
        assert hasattr(rule, 'business_rationale')
        assert hasattr(rule, 'strict')


def test_all_constraint_rules_are_strict(rila_methodology):
    """Test that all RILA constraint rules are strict."""
    rules = rila_methodology.get_constraint_rules()

    for rule in rules:
        assert rule.strict == True, f"Rule {rule.constraint_type} should be strict"


# =============================================================================
# OWN RATE CONSTRAINT TESTS
# =============================================================================


def test_own_rate_constraint_exists(rila_methodology):
    """Test that own rate positive constraint exists."""
    rules = rila_methodology.get_constraint_rules()

    own_rate_rules = [
        rule for rule in rules
        if rule.constraint_type in ['OWN_RATE_POSITIVE', 'OWN_RATE_POSITIVE_P']
    ]

    assert len(own_rate_rules) == 2, "Should have 2 own rate constraints"


def test_own_rate_expected_sign_positive(rila_methodology):
    """Test that own rate constraints expect positive coefficients."""
    rules = rila_methodology.get_constraint_rules()

    for rule in rules:
        if 'OWN_RATE' in rule.constraint_type:
            assert rule.expected_sign == "positive"


def test_own_rate_pattern_matches_prudential(rila_methodology):
    """Test that own rate pattern matches Prudential rate features."""
    rules = rila_methodology.get_constraint_rules()

    prudential_rule = next(
        (rule for rule in rules if rule.constraint_type == 'OWN_RATE_POSITIVE'),
        None
    )

    assert prudential_rule is not None
    pattern = re.compile(prudential_rule.feature_pattern)

    # Should match prudential rate features
    assert pattern.match('prudential_rate_t1')
    assert pattern.match('prudential_rate_current')
    assert pattern.match('prudential_rate_lag_2')


def test_own_rate_pattern_matches_p_prefix(rila_methodology):
    """Test that own rate pattern matches P_ prefix features."""
    rules = rila_methodology.get_constraint_rules()

    p_rule = next(
        (rule for rule in rules if rule.constraint_type == 'OWN_RATE_POSITIVE_P'),
        None
    )

    assert p_rule is not None
    pattern = re.compile(p_rule.feature_pattern)

    # Should match P_ prefixed features
    assert pattern.match('P_rate_t1')
    assert pattern.match('P_cap_rate_current')
    assert pattern.match('P_rate')


def test_own_rate_business_rationale_present(rila_methodology):
    """Test that own rate constraints have business rationale."""
    rules = rila_methodology.get_constraint_rules()

    for rule in rules:
        if 'OWN_RATE' in rule.constraint_type:
            assert rule.business_rationale is not None
            assert len(rule.business_rationale) > 0
            assert 'attract' in rule.business_rationale.lower()


# =============================================================================
# COMPETITOR CONSTRAINT TESTS
# =============================================================================


def test_competitor_constraint_exists(rila_methodology):
    """Test that competitor negative constraint exists."""
    rules = rila_methodology.get_constraint_rules()

    competitor_rules = [
        rule for rule in rules
        if rule.constraint_type in ['COMPETITOR_NEGATIVE', 'COMPETITOR_NEGATIVE_C']
    ]

    assert len(competitor_rules) == 2, "Should have 2 competitor constraints"


def test_competitor_expected_sign_negative(rila_methodology):
    """Test that competitor constraints expect negative coefficients."""
    rules = rila_methodology.get_constraint_rules()

    for rule in rules:
        if 'COMPETITOR_NEGATIVE' in rule.constraint_type:
            assert rule.expected_sign == "negative"


def test_competitor_pattern_matches_competitor_prefix(rila_methodology):
    """Test that competitor pattern matches competitor_ features."""
    rules = rila_methodology.get_constraint_rules()

    competitor_rule = next(
        (rule for rule in rules if rule.constraint_type == 'COMPETITOR_NEGATIVE'),
        None
    )

    assert competitor_rule is not None
    pattern = re.compile(competitor_rule.feature_pattern)

    # Should match competitor_ features
    assert pattern.match('competitor_rate_t1')
    assert pattern.match('competitor_median_t2')
    assert pattern.match('competitor_wink_mean')


def test_competitor_pattern_matches_c_prefix(rila_methodology):
    """Test that competitor pattern matches C_ prefix features."""
    rules = rila_methodology.get_constraint_rules()

    c_rule = next(
        (rule for rule in rules if rule.constraint_type == 'COMPETITOR_NEGATIVE_C'),
        None
    )

    assert c_rule is not None
    pattern = re.compile(c_rule.feature_pattern)

    # Should match C_ prefixed features
    assert pattern.match('C_rate_t1')
    assert pattern.match('C_median_rate_t2')
    assert pattern.match('C_rate')


def test_competitor_business_rationale_present(rila_methodology):
    """Test that competitor constraints have business rationale."""
    rules = rila_methodology.get_constraint_rules()

    for rule in rules:
        if 'COMPETITOR_NEGATIVE' in rule.constraint_type:
            assert rule.business_rationale is not None
            assert len(rule.business_rationale) > 0
            assert 'divert' in rule.business_rationale.lower() or 'away' in rule.business_rationale.lower()


# =============================================================================
# LAG-0 FORBIDDEN CONSTRAINT TESTS
# =============================================================================


def test_lag_zero_forbidden_constraint_exists(rila_methodology):
    """Test that lag-0 forbidden constraint exists."""
    rules = rila_methodology.get_constraint_rules()

    lag_zero_rule = next(
        (rule for rule in rules if rule.constraint_type == 'NO_LAG_ZERO_COMPETITOR'),
        None
    )

    assert lag_zero_rule is not None


def test_lag_zero_expected_sign_forbidden(rila_methodology):
    """Test that lag-0 constraint expects 'forbidden' sign."""
    rules = rila_methodology.get_constraint_rules()

    lag_zero_rule = next(
        (rule for rule in rules if rule.constraint_type == 'NO_LAG_ZERO_COMPETITOR'),
        None
    )

    assert lag_zero_rule.expected_sign == "forbidden"


def test_lag_zero_pattern_matches_t0_suffix(rila_methodology):
    """Test that lag-0 pattern matches _t0 suffix."""
    rules = rila_methodology.get_constraint_rules()

    lag_zero_rule = next(
        (rule for rule in rules if rule.constraint_type == 'NO_LAG_ZERO_COMPETITOR'),
        None
    )

    pattern = re.compile(lag_zero_rule.feature_pattern)

    # Should match _t0 features
    assert pattern.search('competitor_rate_t0')
    assert pattern.search('competitor_median_t0')


def test_lag_zero_pattern_matches_current_suffix(rila_methodology):
    """Test that lag-0 pattern matches _current suffix."""
    rules = rila_methodology.get_constraint_rules()

    lag_zero_rule = next(
        (rule for rule in rules if rule.constraint_type == 'NO_LAG_ZERO_COMPETITOR'),
        None
    )

    pattern = re.compile(lag_zero_rule.feature_pattern)

    # Should match _current features
    assert pattern.search('competitor_rate_current')
    assert pattern.search('competitor_median_current')


def test_lag_zero_pattern_does_not_match_lag_1(rila_methodology):
    """Test that lag-0 pattern does not match lagged features."""
    rules = rila_methodology.get_constraint_rules()

    lag_zero_rule = next(
        (rule for rule in rules if rule.constraint_type == 'NO_LAG_ZERO_COMPETITOR'),
        None
    )

    pattern = re.compile(lag_zero_rule.feature_pattern)

    # Should NOT match lagged features
    assert not pattern.search('competitor_rate_t1')
    assert not pattern.search('competitor_median_t2')
    assert not pattern.search('competitor_rate_lag_1')


def test_lag_zero_business_rationale_mentions_simultaneity(rila_methodology):
    """Test that lag-0 constraint mentions simultaneity bias."""
    rules = rila_methodology.get_constraint_rules()

    lag_zero_rule = next(
        (rule for rule in rules if rule.constraint_type == 'NO_LAG_ZERO_COMPETITOR'),
        None
    )

    rationale = lag_zero_rule.business_rationale.lower()
    assert 'simultaneity' in rationale or 'causal' in rationale


# =============================================================================
# COEFFICIENT SIGNS TESTS
# =============================================================================


def test_get_coefficient_signs_returns_dict(rila_methodology):
    """Test that get_coefficient_signs returns a dictionary."""
    signs = rila_methodology.get_coefficient_signs()
    assert isinstance(signs, dict)


def test_coefficient_signs_includes_prudential(rila_methodology):
    """Test that coefficient signs include prudential pattern."""
    signs = rila_methodology.get_coefficient_signs()
    assert 'prudential_rate' in signs
    assert signs['prudential_rate'] == 'positive'


def test_coefficient_signs_includes_p_prefix(rila_methodology):
    """Test that coefficient signs include P_ pattern."""
    signs = rila_methodology.get_coefficient_signs()
    assert 'P_' in signs
    assert signs['P_'] == 'positive'


def test_coefficient_signs_includes_competitor(rila_methodology):
    """Test that coefficient signs include competitor pattern."""
    signs = rila_methodology.get_coefficient_signs()
    assert 'competitor_' in signs
    assert signs['competitor_'] == 'negative'


def test_coefficient_signs_includes_c_prefix(rila_methodology):
    """Test that coefficient signs include C_ pattern."""
    signs = rila_methodology.get_coefficient_signs()
    assert 'C_' in signs
    assert signs['C_'] == 'negative'


def test_coefficient_signs_count(rila_methodology):
    """Test that coefficient signs has expected number of entries."""
    signs = rila_methodology.get_coefficient_signs()
    assert len(signs) == 4  # prudential, P_, competitor_, C_


# =============================================================================
# LEAKAGE PATTERNS TESTS
# =============================================================================


def test_get_leakage_patterns_returns_list(rila_methodology):
    """Test that get_leakage_patterns returns a list."""
    patterns = rila_methodology.get_leakage_patterns()
    assert isinstance(patterns, list)


def test_leakage_patterns_non_empty(rila_methodology):
    """Test that leakage patterns list is not empty."""
    patterns = rila_methodology.get_leakage_patterns()
    assert len(patterns) > 0


def test_leakage_patterns_includes_t0(rila_methodology):
    """Test that leakage patterns include _t0 suffix."""
    patterns = rila_methodology.get_leakage_patterns()

    t0_pattern = next((p for p in patterns if '_t0' in p), None)
    assert t0_pattern is not None


def test_leakage_patterns_includes_current(rila_methodology):
    """Test that leakage patterns include _current suffix."""
    patterns = rila_methodology.get_leakage_patterns()

    current_pattern = next((p for p in patterns if '_current' in p), None)
    assert current_pattern is not None


def test_leakage_patterns_includes_forward(rila_methodology):
    """Test that leakage patterns include _forward features."""
    patterns = rila_methodology.get_leakage_patterns()

    forward_pattern = next((p for p in patterns if '_forward' in p), None)
    assert forward_pattern is not None


def test_leakage_patterns_includes_future(rila_methodology):
    """Test that leakage patterns include _future features."""
    patterns = rila_methodology.get_leakage_patterns()

    future_pattern = next((p for p in patterns if '_future' in p), None)
    assert future_pattern is not None


def test_leakage_patterns_match_expected_features():
    """Test that leakage patterns match potentially leaky feature names."""
    methodology = RILAMethodology()
    patterns = methodology.get_leakage_patterns()

    # Compile all patterns
    compiled_patterns = [re.compile(p) for p in patterns]

    # Test leaky features
    leaky_features = [
        'sales_t0',
        'rate_current',
        'sales_forward_1',
        'rate_future_3'
    ]

    for feature in leaky_features:
        matched = any(p.search(feature) for p in compiled_patterns)
        assert matched, f"Leakage pattern should match {feature}"


def test_leakage_patterns_do_not_match_safe_features():
    """Test that leakage patterns do not match safe lagged features."""
    methodology = RILAMethodology()
    patterns = methodology.get_leakage_patterns()

    # Compile all patterns
    compiled_patterns = [re.compile(p) for p in patterns]

    # Test safe features
    safe_features = [
        'sales_t1',
        'rate_t2',
        'rate_lag_1',
        'sales_lag_3'
    ]

    for feature in safe_features:
        matched = any(p.search(feature) for p in compiled_patterns)
        assert not matched, f"Leakage pattern should NOT match {feature}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_constraint_rules_cover_all_coefficient_signs(rila_methodology):
    """Test that constraint rules cover all patterns in coefficient signs."""
    rules = rila_methodology.get_constraint_rules()
    signs = rila_methodology.get_coefficient_signs()

    # Extract all feature patterns from rules
    rule_patterns = [rule.feature_pattern for rule in rules if rule.expected_sign != 'forbidden']

    # Check that each sign pattern has a corresponding rule
    for pattern_key in signs.keys():
        # Find if any rule pattern covers this
        covered = any(pattern_key in rule_pattern for rule_pattern in rule_patterns)
        assert covered, f"Pattern '{pattern_key}' should be covered by constraint rules"


def test_all_strict_rules_have_clear_rationale(rila_methodology):
    """Test that all strict rules have clear business rationale."""
    rules = rila_methodology.get_constraint_rules()

    for rule in rules:
        if rule.strict:
            assert rule.business_rationale is not None
            assert len(rule.business_rationale) > 20, f"Rule {rule.constraint_type} needs detailed rationale"


def test_multiple_methodology_instances_identical():
    """Test that multiple RILAMethodology instances produce identical rules."""
    m1 = RILAMethodology()
    m2 = RILAMethodology()

    rules1 = m1.get_constraint_rules()
    rules2 = m2.get_constraint_rules()

    # Should have same number of rules
    assert len(rules1) == len(rules2)

    # Rules should be identical (ConstraintRule is frozen dataclass)
    for r1, r2 in zip(rules1, rules2):
        assert r1.feature_pattern == r2.feature_pattern
        assert r1.expected_sign == r2.expected_sign
        assert r1.constraint_type == r2.constraint_type


def test_constraint_rules_are_immutable(rila_methodology):
    """Test that ConstraintRule instances are immutable (frozen)."""
    rules = rila_methodology.get_constraint_rules()

    if rules:
        rule = rules[0]
        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            rule.expected_sign = "different"


def test_rila_economic_theory_consistency(rila_methodology):
    """Test that RILA rules are consistent with economic theory."""
    rules = rila_methodology.get_constraint_rules()
    signs = rila_methodology.get_coefficient_signs()

    # Own rates should attract (positive)
    assert signs['prudential_rate'] == 'positive'
    assert signs['P_'] == 'positive'

    # Competitor rates should divert (negative)
    assert signs['competitor_'] == 'negative'
    assert signs['C_'] == 'negative'

    # Lag-0 should be forbidden (leakage)
    lag_zero_rule = next(
        (rule for rule in rules if 'LAG_ZERO' in rule.constraint_type),
        None
    )
    assert lag_zero_rule is not None
    assert lag_zero_rule.expected_sign == 'forbidden'
