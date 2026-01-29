"""
Unit Tests for FIA Methodology
===============================

Tests for src/products/fia_methodology.py covering:
- Constraint rule definitions
- Coefficient sign expectations
- Economic rationale validation
- Competitive structure configuration
- Protocol implementation

Target: 90% coverage for fia_methodology.py

Author: Claude Code
Date: 2026-01-29
"""

import re
import pytest
from src.products.fia_methodology import FIAMethodology
from src.core.registry import ConstraintRule


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def fia_methodology():
    """FIA methodology instance for testing."""
    return FIAMethodology()


# =============================================================================
# BASIC PROPERTY TESTS
# =============================================================================


def test_product_type_property(fia_methodology):
    """Test that product_type returns 'fia'."""
    assert fia_methodology.product_type == "fia"


def test_supports_regime_detection(fia_methodology):
    """Test that FIA does not support regime detection."""
    assert fia_methodology.supports_regime_detection() == False


# =============================================================================
# CONSTRAINT RULES TESTS
# =============================================================================


def test_get_constraint_rules_returns_list(fia_methodology):
    """Test that get_constraint_rules returns a list."""
    rules = fia_methodology.get_constraint_rules()
    assert isinstance(rules, list)


def test_get_constraint_rules_non_empty(fia_methodology):
    """Test that constraint rules list is not empty."""
    rules = fia_methodology.get_constraint_rules()
    assert len(rules) > 0


def test_all_constraint_rules_are_constraint_rule_objects(fia_methodology):
    """Test that all rules are ConstraintRule instances."""
    rules = fia_methodology.get_constraint_rules()
    for rule in rules:
        assert isinstance(rule, ConstraintRule)


def test_constraint_rules_count(fia_methodology):
    """Test that FIA has expected number of constraint rules."""
    rules = fia_methodology.get_constraint_rules()
    # Should have 4 rules: own_rate + competitor + top_n + lag-0 forbidden
    assert len(rules) == 4


def test_constraint_rules_have_required_fields(fia_methodology):
    """Test that all constraint rules have required fields."""
    rules = fia_methodology.get_constraint_rules()

    for rule in rules:
        assert hasattr(rule, 'feature_pattern')
        assert hasattr(rule, 'expected_sign')
        assert hasattr(rule, 'constraint_type')
        assert hasattr(rule, 'business_rationale')
        assert hasattr(rule, 'strict')


def test_all_constraint_rules_are_strict(fia_methodology):
    """Test that all FIA constraint rules are strict."""
    rules = fia_methodology.get_constraint_rules()
    for rule in rules:
        assert rule.strict is True


def test_constraint_rule_types_are_unique(fia_methodology):
    """Test that each constraint has unique type identifier."""
    rules = fia_methodology.get_constraint_rules()
    constraint_types = [rule.constraint_type for rule in rules]
    assert len(constraint_types) == len(set(constraint_types))


# =============================================================================
# CONSTRAINT RULE PATTERN MATCHING TESTS
# =============================================================================


def test_own_rate_pattern_matches_own_rate(fia_methodology):
    """Test that own_rate pattern matches 'own_rate' feature."""
    rules = fia_methodology.get_constraint_rules()
    own_rate_rules = [r for r in rules if r.constraint_type == "OWN_RATE_POSITIVE"]

    assert len(own_rate_rules) == 1
    pattern = own_rate_rules[0].feature_pattern

    # Test pattern matching
    assert re.match(pattern, "own_rate_lag_1")
    assert re.match(pattern, "prudential_rate_current")
    assert re.match(pattern, "P_cap_rate")


def test_competitor_pattern_matches_competitor(fia_methodology):
    """Test that competitor pattern matches competitor features."""
    rules = fia_methodology.get_constraint_rules()
    competitor_rules = [r for r in rules if r.constraint_type == "COMPETITOR_NEGATIVE"]

    assert len(competitor_rules) == 1
    pattern = competitor_rules[0].feature_pattern

    # Test pattern matching
    assert re.match(pattern, "competitor_mean")
    assert re.match(pattern, "C_cap_rate")  # Must end with 'rate' per pattern


def test_top_n_pattern_matches_top_n(fia_methodology):
    """Test that top_n pattern matches top-N features."""
    rules = fia_methodology.get_constraint_rules()
    top_n_rules = [r for r in rules if r.constraint_type == "TOP_N_NEGATIVE"]

    assert len(top_n_rules) == 1
    pattern = top_n_rules[0].feature_pattern

    # Test pattern matching
    assert re.match(pattern, "top5_mean")
    assert re.match(pattern, "top3_rate")


def test_lag_zero_pattern_matches_lag_zero(fia_methodology):
    """Test that lag-0 pattern matches lag-0 features."""
    rules = fia_methodology.get_constraint_rules()
    lag_zero_rules = [r for r in rules if r.constraint_type == "NO_LAG_ZERO_COMPETITOR"]

    assert len(lag_zero_rules) == 1
    pattern = lag_zero_rules[0].feature_pattern

    # Test pattern matching
    assert re.match(pattern, "competitor_mean_t0")
    assert re.match(pattern, "competitor_rate_current")


# =============================================================================
# EXPECTED SIGN TESTS
# =============================================================================


def test_own_rate_expected_sign_is_positive(fia_methodology):
    """Test that own rate rules expect positive coefficient."""
    rules = fia_methodology.get_constraint_rules()
    own_rate_rules = [r for r in rules if r.constraint_type == "OWN_RATE_POSITIVE"]

    assert own_rate_rules[0].expected_sign == "positive"


def test_competitor_expected_sign_is_negative(fia_methodology):
    """Test that competitor rules expect negative coefficient."""
    rules = fia_methodology.get_constraint_rules()
    competitor_rules = [r for r in rules if r.constraint_type == "COMPETITOR_NEGATIVE"]

    assert competitor_rules[0].expected_sign == "negative"


def test_top_n_expected_sign_is_negative(fia_methodology):
    """Test that top-N rules expect negative coefficient."""
    rules = fia_methodology.get_constraint_rules()
    top_n_rules = [r for r in rules if r.constraint_type == "TOP_N_NEGATIVE"]

    assert top_n_rules[0].expected_sign == "negative"


def test_lag_zero_expected_sign_is_forbidden(fia_methodology):
    """Test that lag-0 rules expect forbidden (exclusion)."""
    rules = fia_methodology.get_constraint_rules()
    lag_zero_rules = [r for r in rules if r.constraint_type == "NO_LAG_ZERO_COMPETITOR"]

    assert lag_zero_rules[0].expected_sign == "forbidden"


# =============================================================================
# COEFFICIENT SIGNS TESTS
# =============================================================================


def test_get_coefficient_signs_returns_dict(fia_methodology):
    """Test that get_coefficient_signs returns a dict."""
    signs = fia_methodology.get_coefficient_signs()
    assert isinstance(signs, dict)


def test_get_coefficient_signs_non_empty(fia_methodology):
    """Test that coefficient signs dict is not empty."""
    signs = fia_methodology.get_coefficient_signs()
    assert len(signs) > 0


def test_get_coefficient_signs_own_rate_positive(fia_methodology):
    """Test that own_rate pattern expects positive."""
    signs = fia_methodology.get_coefficient_signs()
    assert signs["own_rate"] == "positive"
    assert signs["prudential_rate"] == "positive"
    assert signs["P_"] == "positive"


def test_get_coefficient_signs_competitor_negative(fia_methodology):
    """Test that competitor patterns expect negative."""
    signs = fia_methodology.get_coefficient_signs()
    assert signs["competitor_"] == "negative"
    assert signs["C_"] == "negative"
    assert signs["top"] == "negative"


# =============================================================================
# COMPETITIVE STRUCTURE TESTS
# =============================================================================


def test_get_competitive_structure_returns_dict(fia_methodology):
    """Test that get_competitive_structure returns a dict."""
    structure = fia_methodology.get_competitive_structure()
    assert isinstance(structure, dict)


def test_get_competitive_structure_has_aggregation_method(fia_methodology):
    """Test that competitive structure specifies aggregation method."""
    structure = fia_methodology.get_competitive_structure()
    assert "aggregation_method" in structure
    assert structure["aggregation_method"] == "top_n"


def test_get_competitive_structure_has_n_competitors(fia_methodology):
    """Test that competitive structure specifies number of competitors."""
    structure = fia_methodology.get_competitive_structure()
    assert "n_competitors" in structure


def test_get_competitive_structure_has_rationale(fia_methodology):
    """Test that competitive structure includes rationale."""
    structure = fia_methodology.get_competitive_structure()
    assert "rationale" in structure
    assert len(structure["rationale"]) > 20  # Should be descriptive


# =============================================================================
# BUSINESS RATIONALE TESTS
# =============================================================================


def test_all_rules_have_business_rationale(fia_methodology):
    """Test that all constraint rules have business rationale."""
    rules = fia_methodology.get_constraint_rules()

    for rule in rules:
        assert rule.business_rationale is not None
        assert len(rule.business_rationale) > 20  # Should be meaningful


def test_business_rationale_mentions_fia_concepts(fia_methodology):
    """Test that business rationale mentions FIA-relevant concepts."""
    rules = fia_methodology.get_constraint_rules()

    # Collect all rationales
    rationales = " ".join([rule.business_rationale for rule in rules])

    # Should mention FIA-relevant concepts
    fia_concepts = ["rate", "customer", "sales", "competitor", "FIA", "market"]
    assert any(concept.lower() in rationales.lower() for concept in fia_concepts)


# =============================================================================
# COMPARISON WITH RILA TESTS
# =============================================================================


def test_fia_has_fewer_rules_than_rila():
    """Test that FIA has fewer rules than RILA (no buffer-specific rules)."""
    from src.products.rila_methodology import RILAMethodology

    fia = FIAMethodology()
    rila = RILAMethodology()

    fia_rules = fia.get_constraint_rules()
    rila_rules = rila.get_constraint_rules()

    # FIA has no buffer rules, so should have fewer total rules
    assert len(fia_rules) < len(rila_rules)


def test_fia_shares_core_rules_with_rila():
    """Test that FIA and RILA share core economic rules."""
    from src.products.rila_methodology import RILAMethodology

    fia = FIAMethodology()
    rila = RILAMethodology()

    fia_constraint_types = {r.constraint_type for r in fia.get_constraint_rules()}
    rila_constraint_types = {r.constraint_type for r in rila.get_constraint_rules()}

    # Should share core constraint types
    core_constraints = {"OWN_RATE_POSITIVE", "COMPETITOR_NEGATIVE", "NO_LAG_ZERO_COMPETITOR"}
    assert core_constraints.issubset(fia_constraint_types)
    assert core_constraints.issubset(rila_constraint_types)


def test_fia_uses_top_n_rila_may_use_weighted():
    """Test that FIA prefers top-N aggregation."""
    fia = FIAMethodology()

    structure = fia.get_competitive_structure()
    assert structure["aggregation_method"] == "top_n"


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


def test_implements_product_methodology_protocol():
    """Test that FIAMethodology implements ProductMethodology protocol."""
    from src.products.base import ProductMethodology

    fia = FIAMethodology()

    # Check protocol methods exist
    assert hasattr(fia, 'get_constraint_rules')
    assert hasattr(fia, 'get_coefficient_signs')
    assert hasattr(fia, 'supports_regime_detection')
    assert hasattr(fia, 'product_type')


def test_protocol_methods_are_callable():
    """Test that protocol methods are callable."""
    fia = FIAMethodology()

    assert callable(fia.get_constraint_rules)
    assert callable(fia.get_coefficient_signs)
    assert callable(fia.supports_regime_detection)
