"""
Unit Tests for Products Base Module
===================================

Tests for src/products/base.py covering:
- ProductMethodology Protocol definition
- ConstraintRule re-export
- Protocol implementation by concrete methodologies

Target: Verify Protocol interface and re-exports work correctly

Author: Claude Code
Date: 2026-01-31
"""

import pytest

from src.products.base import ProductMethodology, ConstraintRule
from src.products.rila_methodology import RILAMethodology
from src.products.fia_methodology import FIAMethodology
from src.products.myga_methodology import MYGAMethodology


# =============================================================================
# PROTOCOL DEFINITION TESTS
# =============================================================================


class TestProductMethodologyProtocol:
    """Tests for ProductMethodology Protocol definition."""

    def test_protocol_has_get_constraint_rules_method(self):
        """Protocol should define get_constraint_rules method."""
        assert hasattr(ProductMethodology, "get_constraint_rules")

    def test_protocol_has_get_coefficient_signs_method(self):
        """Protocol should define get_coefficient_signs method."""
        assert hasattr(ProductMethodology, "get_coefficient_signs")

    def test_protocol_has_supports_regime_detection_method(self):
        """Protocol should define supports_regime_detection method."""
        assert hasattr(ProductMethodology, "supports_regime_detection")

    def test_protocol_has_product_type_property(self):
        """Protocol should define product_type property."""
        assert hasattr(ProductMethodology, "product_type")


# =============================================================================
# CONSTRAINT RULE RE-EXPORT TESTS
# =============================================================================


class TestConstraintRuleReExport:
    """Tests for ConstraintRule re-export from base.py."""

    def test_constraint_rule_imported_from_base(self):
        """ConstraintRule should be importable from base.py."""
        assert ConstraintRule is not None

    def test_constraint_rule_is_same_as_registry(self):
        """ConstraintRule should be same class as in registry."""
        from src.core.registry import ConstraintRule as RegistryConstraintRule

        assert ConstraintRule is RegistryConstraintRule

    def test_constraint_rule_can_be_instantiated(self):
        """ConstraintRule should be instantiatable."""
        rule = ConstraintRule(
            feature_pattern="test_.*",
            expected_sign="positive",
            constraint_type="TEST_CONSTRAINT",
            business_rationale="Test rationale",
            strict=True
        )
        assert rule.feature_pattern == "test_.*"
        assert rule.expected_sign == "positive"
        assert rule.constraint_type == "TEST_CONSTRAINT"
        assert rule.business_rationale == "Test rationale"
        assert rule.strict is True


# =============================================================================
# MODULE EXPORT TESTS
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_correct(self):
        """Test that __all__ contains expected exports."""
        from src.products.base import __all__

        assert "ProductMethodology" in __all__
        assert "ConstraintRule" in __all__
        assert len(__all__) == 2


# =============================================================================
# CONCRETE IMPLEMENTATION TESTS
# =============================================================================


class TestConcreteImplementations:
    """Tests that concrete methodologies implement the Protocol correctly."""

    @pytest.fixture(params=[RILAMethodology, FIAMethodology, MYGAMethodology])
    def methodology_class(self, request):
        """Parametrized fixture for all methodology classes."""
        return request.param

    def test_concrete_has_get_constraint_rules(self, methodology_class):
        """All concrete classes should have get_constraint_rules method."""
        methodology = methodology_class()
        assert hasattr(methodology, "get_constraint_rules")
        rules = methodology.get_constraint_rules()
        assert isinstance(rules, list)

    def test_concrete_has_get_coefficient_signs(self, methodology_class):
        """All concrete classes should have get_coefficient_signs method."""
        methodology = methodology_class()
        assert hasattr(methodology, "get_coefficient_signs")
        signs = methodology.get_coefficient_signs()
        assert isinstance(signs, dict)

    def test_concrete_has_supports_regime_detection(self, methodology_class):
        """All concrete classes should have supports_regime_detection method."""
        methodology = methodology_class()
        assert hasattr(methodology, "supports_regime_detection")
        result = methodology.supports_regime_detection()
        assert isinstance(result, bool)

    def test_concrete_has_product_type_property(self, methodology_class):
        """All concrete classes should have product_type property."""
        methodology = methodology_class()
        assert hasattr(methodology, "product_type")
        product_type = methodology.product_type
        assert isinstance(product_type, str)
        assert product_type in ["rila", "fia", "myga"]


# =============================================================================
# PROTOCOL STRUCTURAL CONFORMANCE TESTS
# =============================================================================


class TestProtocolConformance:
    """Tests for structural conformance with Protocol."""

    def test_rila_returns_constraint_rules(self):
        """RILA methodology returns ConstraintRule objects."""
        methodology = RILAMethodology()
        rules = methodology.get_constraint_rules()

        for rule in rules:
            assert isinstance(rule, ConstraintRule)

    def test_fia_returns_constraint_rules(self):
        """FIA methodology returns ConstraintRule objects."""
        methodology = FIAMethodology()
        rules = methodology.get_constraint_rules()

        for rule in rules:
            assert isinstance(rule, ConstraintRule)

    def test_myga_returns_constraint_rules(self):
        """MYGA methodology returns ConstraintRule objects."""
        methodology = MYGAMethodology()
        rules = methodology.get_constraint_rules()

        for rule in rules:
            assert isinstance(rule, ConstraintRule)

    def test_all_methodologies_have_consistent_product_types(self):
        """Each methodology should have a unique product type."""
        rila = RILAMethodology()
        fia = FIAMethodology()
        myga = MYGAMethodology()

        product_types = {rila.product_type, fia.product_type, myga.product_type}
        assert len(product_types) == 3  # All unique
        assert product_types == {"rila", "fia", "myga"}

    def test_all_methodologies_have_valid_coefficient_signs(self):
        """All methodologies should have valid coefficient sign values."""
        valid_signs = {"positive", "negative"}

        for MethodologyClass in [RILAMethodology, FIAMethodology, MYGAMethodology]:
            methodology = MethodologyClass()
            signs = methodology.get_coefficient_signs()

            for pattern, sign in signs.items():
                assert sign in valid_signs, \
                    f"{MethodologyClass.__name__} has invalid sign '{sign}' for '{pattern}'"
