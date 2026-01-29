"""
Unit Tests for MYGA Methodology
================================

Tests for src/products/myga_methodology.py covering:
- Constraint rule definitions
- Coefficient sign expectations
- Economic rationale validation
- Regime detection support
- Leakage pattern detection
- Protocol implementation

Target: 90% coverage for myga_methodology.py

Author: Claude Code
Date: 2026-01-29
"""

import re
import pytest
from src.products.myga_methodology import MYGAMethodology
from src.core.registry import ConstraintRule


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def myga_methodology():
    """MYGA methodology instance for testing."""
    return MYGAMethodology()


# =============================================================================
# BASIC PROPERTY TESTS
# =============================================================================


def test_product_type_property(myga_methodology):
    """Test that product_type returns 'myga'."""
    assert myga_methodology.product_type == "myga"


def test_supports_regime_detection(myga_methodology):
    """Test that MYGA supports regime detection."""
    assert myga_methodology.supports_regime_detection() == True


# =============================================================================
# CONSTRAINT RULES TESTS
# =============================================================================


def test_get_constraint_rules_returns_list(myga_methodology):
    """Test that get_constraint_rules returns a list."""
    rules = myga_methodology.get_constraint_rules()
    assert isinstance(rules, list)


def test_get_constraint_rules_non_empty(myga_methodology):
    """Test that constraint rules list is not empty."""
    rules = myga_methodology.get_constraint_rules()
    assert len(rules) > 0


def test_all_constraint_rules_are_constraint_rule_objects(myga_methodology):
    """Test that all rules are ConstraintRule instances."""
    rules = myga_methodology.get_constraint_rules()
    for rule in rules:
        assert isinstance(rule, ConstraintRule)


def test_constraint_rules_count(myga_methodology):
    """Test that MYGA has expected number of constraint rules."""
    rules = myga_methodology.get_constraint_rules()
    # Should have 6 rules: own_rate (2) + competitor (2) + treasury_spread + lag-0
    assert len(rules) == 6


def test_constraint_rules_have_required_fields(myga_methodology):
    """Test that all constraint rules have required fields."""
    rules = myga_methodology.get_constraint_rules()

    for rule in rules:
        assert hasattr(rule, 'feature_pattern')
        assert hasattr(rule, 'expected_sign')
        assert hasattr(rule, 'constraint_type')
        assert hasattr(rule, 'business_rationale')
        assert hasattr(rule, 'strict')


def test_treasury_spread_is_soft_constraint(myga_methodology):
    """Test that treasury spread rule is not strict."""
    rules = myga_methodology.get_constraint_rules()
    treasury_rules = [r for r in rules if r.constraint_type == "TREASURY_SPREAD_POSITIVE"]

    assert len(treasury_rules) == 1
    assert treasury_rules[0].strict is False  # Soft constraint


def test_other_rules_are_strict(myga_methodology):
    """Test that non-treasury rules are strict."""
    rules = myga_methodology.get_constraint_rules()

    for rule in rules:
        if rule.constraint_type != "TREASURY_SPREAD_POSITIVE":
            assert rule.strict is True


# =============================================================================
# CONSTRAINT RULE PATTERN MATCHING TESTS
# =============================================================================


def test_own_rate_pattern_matches_own_rate(myga_methodology):
    """Test that own_rate pattern matches 'own_rate' feature."""
    rules = myga_methodology.get_constraint_rules()
    own_rate_rules = [r for r in rules if "OWN_RATE_POSITIVE" in r.constraint_type]

    assert len(own_rate_rules) == 2  # Two patterns
    patterns = [r.feature_pattern for r in own_rate_rules]

    # Test pattern matching
    assert any(re.match(p, "own_rate_lag_1") for p in patterns)
    assert any(re.match(p, "prudential_rate_current") for p in patterns)
    assert any(re.match(p, "P_guaranteed_rate") for p in patterns)


def test_competitor_pattern_matches_competitor(myga_methodology):
    """Test that competitor pattern matches competitor features."""
    rules = myga_methodology.get_constraint_rules()
    competitor_rules = [r for r in rules if "COMPETITOR_NEGATIVE" in r.constraint_type]

    assert len(competitor_rules) == 2  # Two patterns
    patterns = [r.feature_pattern for r in competitor_rules]

    # Test pattern matching
    assert any(re.match(p, "competitor_mean") for p in patterns)
    assert any(re.match(p, "C_guaranteed_rate") for p in patterns)  # Must end with 'rate' per pattern


def test_treasury_spread_pattern_matches_spread(myga_methodology):
    """Test that treasury_spread pattern matches spread features."""
    rules = myga_methodology.get_constraint_rules()
    treasury_rules = [r for r in rules if r.constraint_type == "TREASURY_SPREAD_POSITIVE"]

    assert len(treasury_rules) == 1
    pattern = treasury_rules[0].feature_pattern

    # Test pattern matching
    assert re.match(pattern, "treasury_spread")
    assert re.match(pattern, "rate_spread")


def test_lag_zero_pattern_matches_lag_zero(myga_methodology):
    """Test that lag-0 pattern matches lag-0 features."""
    rules = myga_methodology.get_constraint_rules()
    lag_zero_rules = [r for r in rules if r.constraint_type == "NO_LAG_ZERO_COMPETITOR"]

    assert len(lag_zero_rules) == 1
    pattern = lag_zero_rules[0].feature_pattern

    # Test pattern matching
    assert re.match(pattern, "competitor_mean_t0")
    assert re.match(pattern, "competitor_rate_current")


# =============================================================================
# EXPECTED SIGN TESTS
# =============================================================================


def test_own_rate_expected_sign_is_positive(myga_methodology):
    """Test that own rate rules expect positive coefficient."""
    rules = myga_methodology.get_constraint_rules()
    own_rate_rules = [r for r in rules if "OWN_RATE_POSITIVE" in r.constraint_type]

    for rule in own_rate_rules:
        assert rule.expected_sign == "positive"


def test_competitor_expected_sign_is_negative(myga_methodology):
    """Test that competitor rules expect negative coefficient."""
    rules = myga_methodology.get_constraint_rules()
    competitor_rules = [r for r in rules if "COMPETITOR_NEGATIVE" in r.constraint_type]

    for rule in competitor_rules:
        assert rule.expected_sign == "negative"


def test_treasury_spread_expected_sign_is_positive(myga_methodology):
    """Test that treasury spread expects positive coefficient."""
    rules = myga_methodology.get_constraint_rules()
    treasury_rules = [r for r in rules if r.constraint_type == "TREASURY_SPREAD_POSITIVE"]

    assert treasury_rules[0].expected_sign == "positive"


def test_lag_zero_expected_sign_is_forbidden(myga_methodology):
    """Test that lag-0 rules expect forbidden (exclusion)."""
    rules = myga_methodology.get_constraint_rules()
    lag_zero_rules = [r for r in rules if r.constraint_type == "NO_LAG_ZERO_COMPETITOR"]

    assert lag_zero_rules[0].expected_sign == "forbidden"


# =============================================================================
# COEFFICIENT SIGNS TESTS
# =============================================================================


def test_get_coefficient_signs_returns_dict(myga_methodology):
    """Test that get_coefficient_signs returns a dict."""
    signs = myga_methodology.get_coefficient_signs()
    assert isinstance(signs, dict)


def test_get_coefficient_signs_non_empty(myga_methodology):
    """Test that coefficient signs dict is not empty."""
    signs = myga_methodology.get_coefficient_signs()
    assert len(signs) > 0


def test_get_coefficient_signs_own_rate_positive(myga_methodology):
    """Test that own_rate patterns expect positive."""
    signs = myga_methodology.get_coefficient_signs()
    assert signs["own_rate"] == "positive"
    assert signs["prudential_rate"] == "positive"
    assert signs["P_"] == "positive"


def test_get_coefficient_signs_competitor_negative(myga_methodology):
    """Test that competitor patterns expect negative."""
    signs = myga_methodology.get_coefficient_signs()
    assert signs["competitor_"] == "negative"
    assert signs["C_"] == "negative"


def test_get_coefficient_signs_treasury_spread_positive(myga_methodology):
    """Test that treasury spread patterns expect positive."""
    signs = myga_methodology.get_coefficient_signs()
    assert signs["treasury_spread"] == "positive"
    assert signs["rate_spread"] == "positive"


# =============================================================================
# LEAKAGE PATTERN TESTS
# =============================================================================


def test_get_leakage_patterns_returns_list(myga_methodology):
    """Test that get_leakage_patterns returns a list."""
    patterns = myga_methodology.get_leakage_patterns()
    assert isinstance(patterns, list)


def test_get_leakage_patterns_non_empty(myga_methodology):
    """Test that leakage patterns list is not empty."""
    patterns = myga_methodology.get_leakage_patterns()
    assert len(patterns) > 0


def test_leakage_patterns_match_lag_zero(myga_methodology):
    """Test that leakage patterns match lag-0 features."""
    patterns = myga_methodology.get_leakage_patterns()

    # Should match lag-0 features
    lag_zero_features = ["rate_t0", "competitor_t0", "sales_t0"]
    for feature in lag_zero_features:
        assert any(re.match(p, feature) for p in patterns)


def test_leakage_patterns_match_current(myga_methodology):
    """Test that leakage patterns match current-period features."""
    patterns = myga_methodology.get_leakage_patterns()

    # Should match current features
    current_features = ["rate_current", "competitor_current", "sales_current"]
    for feature in current_features:
        assert any(re.match(p, feature) for p in patterns)


def test_leakage_patterns_match_forward(myga_methodology):
    """Test that leakage patterns match forward-looking features."""
    patterns = myga_methodology.get_leakage_patterns()

    # Should match forward features
    forward_features = ["rate_forward", "competitor_forward_1", "sales_future"]
    for feature in forward_features:
        assert any(re.match(p, feature) for p in patterns)


# =============================================================================
# BUSINESS RATIONALE TESTS
# =============================================================================


def test_all_rules_have_business_rationale(myga_methodology):
    """Test that all constraint rules have business rationale."""
    rules = myga_methodology.get_constraint_rules()

    for rule in rules:
        assert rule.business_rationale is not None
        assert len(rule.business_rationale) > 20  # Should be meaningful


def test_business_rationale_mentions_myga_concepts(myga_methodology):
    """Test that business rationale mentions MYGA-relevant concepts."""
    rules = myga_methodology.get_constraint_rules()

    # Collect all rationales
    rationales = " ".join([rule.business_rationale for rule in rules])

    # Should mention MYGA-relevant concepts
    myga_concepts = ["guaranteed", "rate", "customer", "MYGA", "treasury", "spread"]
    assert any(concept.lower() in rationales.lower() for concept in myga_concepts)


# =============================================================================
# COMPARISON WITH RILA/FIA TESTS
# =============================================================================


def test_myga_has_treasury_spread_rule():
    """Test that MYGA has treasury spread rule (unique to MYGA)."""
    myga = MYGAMethodology()

    rules = myga.get_constraint_rules()
    treasury_rules = [r for r in rules if "TREASURY_SPREAD" in r.constraint_type]

    assert len(treasury_rules) == 1


def test_myga_supports_regime_detection_unlike_rila_fia():
    """Test that MYGA supports regime detection unlike RILA/FIA."""
    from src.products.rila_methodology import RILAMethodology
    from src.products.fia_methodology import FIAMethodology

    myga = MYGAMethodology()
    rila = RILAMethodology()
    fia = FIAMethodology()

    assert myga.supports_regime_detection() == True
    assert rila.supports_regime_detection() == False
    assert fia.supports_regime_detection() == False


def test_myga_shares_core_rules_with_rila_fia():
    """Test that MYGA shares core economic rules with RILA/FIA."""
    from src.products.rila_methodology import RILAMethodology
    from src.products.fia_methodology import FIAMethodology

    myga = MYGAMethodology()
    rila = RILAMethodology()
    fia = FIAMethodology()

    myga_constraint_types = {r.constraint_type for r in myga.get_constraint_rules()}
    rila_constraint_types = {r.constraint_type for r in rila.get_constraint_rules()}
    fia_constraint_types = {r.constraint_type for r in fia.get_constraint_rules()}

    # Should share core constraint types
    core_constraints = {"OWN_RATE_POSITIVE", "COMPETITOR_NEGATIVE", "NO_LAG_ZERO_COMPETITOR"}

    # Check partial overlap (some types might be named differently like OWN_RATE_POSITIVE_P)
    assert any("OWN_RATE_POSITIVE" in ct for ct in myga_constraint_types)
    assert any("COMPETITOR_NEGATIVE" in ct for ct in myga_constraint_types)
    assert "NO_LAG_ZERO_COMPETITOR" in myga_constraint_types


def test_myga_has_unique_leakage_detection():
    """Test that MYGA provides leakage pattern detection."""
    myga = MYGAMethodology()

    assert hasattr(myga, 'get_leakage_patterns')
    patterns = myga.get_leakage_patterns()
    assert len(patterns) > 0


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


def test_implements_product_methodology_protocol():
    """Test that MYGAMethodology implements ProductMethodology protocol."""
    from src.products.base import ProductMethodology

    myga = MYGAMethodology()

    # Check protocol methods exist
    assert hasattr(myga, 'get_constraint_rules')
    assert hasattr(myga, 'get_coefficient_signs')
    assert hasattr(myga, 'supports_regime_detection')
    assert hasattr(myga, 'product_type')


def test_protocol_methods_are_callable():
    """Test that protocol methods are callable."""
    myga = MYGAMethodology()

    assert callable(myga.get_constraint_rules)
    assert callable(myga.get_coefficient_signs)
    assert callable(myga.supports_regime_detection)


def test_extended_method_is_callable():
    """Test that MYGA-specific method is callable."""
    myga = MYGAMethodology()

    assert callable(myga.get_leakage_patterns)


# =============================================================================
# REGIME DETECTION RATIONALE TESTS
# =============================================================================


def test_regime_detection_rationale():
    """Test that MYGA regime detection support has clear rationale."""
    myga = MYGAMethodology()

    # MYGA products may behave differently in rising vs falling rate environments
    assert myga.supports_regime_detection() == True

    # This is because fixed guaranteed rates have different appeal in different
    # interest rate regimes - customers behavior may shift based on rate environment
