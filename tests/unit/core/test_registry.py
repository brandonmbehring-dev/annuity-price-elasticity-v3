"""
Tests for src.core.registry module.

Tests registry classes and ConstraintRule dataclass.
"""

import pytest

from src.core.registry import (
    ConstraintRule,
    BusinessRulesRegistry,
    AdapterRegistry,
    AggregationRegistry,
    get_methodology,
    get_adapter,
    get_aggregation_strategy,
)


class TestConstraintRule:
    """Tests for ConstraintRule dataclass."""

    def test_constraint_rule_creation(self):
        """ConstraintRule should be created with required fields."""
        rule = ConstraintRule(
            feature_pattern="competitor_",
            expected_sign="negative",
            constraint_type="COMPETITOR_NEGATIVE",
            business_rationale="Higher competitor rates attract customers away",
        )
        assert rule.feature_pattern == "competitor_"
        assert rule.expected_sign == "negative"
        assert rule.constraint_type == "COMPETITOR_NEGATIVE"
        assert rule.strict is True  # Default

    def test_constraint_rule_strict_default(self):
        """ConstraintRule strict should default to True."""
        rule = ConstraintRule(
            feature_pattern="test_",
            expected_sign="positive",
            constraint_type="TEST",
            business_rationale="Test rule",
        )
        assert rule.strict is True

    def test_constraint_rule_strict_override(self):
        """ConstraintRule strict can be set to False."""
        rule = ConstraintRule(
            feature_pattern="test_",
            expected_sign="positive",
            constraint_type="TEST",
            business_rationale="Test rule",
            strict=False,
        )
        assert rule.strict is False

    def test_constraint_rule_is_frozen(self):
        """ConstraintRule should be immutable (frozen)."""
        rule = ConstraintRule(
            feature_pattern="test_",
            expected_sign="positive",
            constraint_type="TEST",
            business_rationale="Test rule",
        )
        with pytest.raises(AttributeError):
            rule.feature_pattern = "modified_"

    def test_constraint_rule_equality(self):
        """Two ConstraintRules with same values should be equal."""
        rule1 = ConstraintRule(
            feature_pattern="test_",
            expected_sign="positive",
            constraint_type="TEST",
            business_rationale="Test rule",
        )
        rule2 = ConstraintRule(
            feature_pattern="test_",
            expected_sign="positive",
            constraint_type="TEST",
            business_rationale="Test rule",
        )
        assert rule1 == rule2


class TestBusinessRulesRegistry:
    """Tests for BusinessRulesRegistry class."""

    def test_get_rila_methodology(self):
        """Registry should return RILA methodology."""
        methodology = BusinessRulesRegistry.get("rila")
        assert methodology.product_type == "rila"

    def test_get_fia_methodology(self):
        """Registry should return FIA methodology."""
        methodology = BusinessRulesRegistry.get("fia")
        assert methodology.product_type == "fia"

    def test_get_invalid_methodology_raises(self):
        """Registry should raise KeyError for invalid product type."""
        with pytest.raises(KeyError) as exc_info:
            BusinessRulesRegistry.get("invalid_product")
        assert "invalid_product" in str(exc_info.value)

    def test_list_registered_includes_rila_fia(self):
        """Registry should list at least rila and fia."""
        registered = BusinessRulesRegistry.list_registered()
        assert "rila" in registered
        assert "fia" in registered

    def test_get_constraint_rules_returns_list(self):
        """Registry should return constraint rules as list."""
        rules = BusinessRulesRegistry.get_constraint_rules("rila")
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_constraint_rules_are_constraint_rule_instances(self):
        """Constraint rules should be ConstraintRule instances."""
        rules = BusinessRulesRegistry.get_constraint_rules("rila")
        for rule in rules:
            assert isinstance(rule, ConstraintRule)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_methodology_wrapper(self):
        """get_methodology should delegate to BusinessRulesRegistry."""
        methodology = get_methodology("rila")
        assert methodology.product_type == "rila"

    def test_get_methodology_invalid_raises(self):
        """get_methodology should raise ValueError for invalid type."""
        with pytest.raises(ValueError, match="Unknown product_type"):
            get_methodology("invalid")


class TestAggregationRegistry:
    """Tests for AggregationRegistry class."""

    def test_list_registered_strategies(self):
        """Registry should list available strategies."""
        strategies = AggregationRegistry._factories
        # After initialization, should have some registered
        AggregationRegistry._ensure_initialized()
        assert len(AggregationRegistry._factories) > 0

    def test_get_for_product_type_rila(self):
        """Should get weighted strategy for RILA."""
        strategy = AggregationRegistry.get_for_product_type("rila")
        # Strategy should be returned (implementation details may vary)
        assert strategy is not None

    def test_get_for_product_type_invalid_raises(self):
        """Should raise for invalid product type."""
        with pytest.raises(KeyError):
            AggregationRegistry.get_for_product_type("invalid_type")
