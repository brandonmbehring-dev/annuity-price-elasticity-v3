"""
Unit tests for src/core/product_registry.py

Tests product lookup, validation, and registry functions.
"""

import pytest


class TestProductDefinition:
    """Tests for ProductDefinition dataclass."""

    def test_creates_rila_product(self):
        """Creates RILA product with all fields."""
        from src.core.product_registry import ProductDefinition

        product = ProductDefinition(
            code="TEST",
            name="Test Product",
            product_type="rila",
            fixture_name="Test fixture",
            buffer_rate=20,
            term_years=6,
            aliases=("Alias1", "Alias2")
        )

        assert product.code == "TEST"
        assert product.name == "Test Product"
        assert product.product_type == "rila"
        assert product.buffer_rate == 20
        assert product.term_years == 6
        assert len(product.aliases) == 2

    def test_creates_fia_product_without_buffer(self):
        """Creates FIA product with None buffer_rate."""
        from src.core.product_registry import ProductDefinition

        product = ProductDefinition(
            code="FIA_TEST",
            name="FIA Test",
            product_type="fia",
            fixture_name="FIA fixture",
            buffer_rate=None,
            term_years=5
        )

        assert product.buffer_rate is None
        assert product.product_type == "fia"

    def test_frozen_dataclass_is_immutable(self):
        """ProductDefinition is frozen/immutable."""
        from src.core.product_registry import ProductDefinition

        product = ProductDefinition(
            code="TEST",
            name="Test",
            product_type="rila",
            fixture_name="Fixture",
            buffer_rate=10,
            term_years=6
        )

        with pytest.raises(AttributeError):
            product.code = "NEW_CODE"

    def test_default_empty_aliases(self):
        """Default aliases is empty tuple."""
        from src.core.product_registry import ProductDefinition

        product = ProductDefinition(
            code="TEST",
            name="Test",
            product_type="rila",
            fixture_name="Fixture",
            buffer_rate=10,
            term_years=6
        )

        assert product.aliases == ()


class TestProductByCodeRegistry:
    """Tests for PRODUCT_BY_CODE registry."""

    def test_contains_6y20b(self):
        """Registry contains 6Y20B product."""
        from src.core.product_registry import PRODUCT_BY_CODE

        assert "6Y20B" in PRODUCT_BY_CODE
        assert PRODUCT_BY_CODE["6Y20B"].code == "6Y20B"

    def test_contains_1y10b(self):
        """Registry contains 1Y10B product."""
        from src.core.product_registry import PRODUCT_BY_CODE

        assert "1Y10B" in PRODUCT_BY_CODE
        assert PRODUCT_BY_CODE["1Y10B"].term_years == 1

    def test_contains_fia_products(self):
        """Registry contains FIA products."""
        from src.core.product_registry import PRODUCT_BY_CODE

        assert "FIA5YR" in PRODUCT_BY_CODE
        assert PRODUCT_BY_CODE["FIA5YR"].product_type == "fia"

    def test_all_products_have_required_fields(self):
        """All products have required fields populated."""
        from src.core.product_registry import PRODUCT_BY_CODE

        for code, product in PRODUCT_BY_CODE.items():
            assert product.code == code
            assert product.name is not None
            assert product.product_type in ("rila", "fia", "myga")
            assert product.fixture_name is not None
            assert product.term_years > 0


class TestProductByNameRegistry:
    """Tests for PRODUCT_BY_NAME registry."""

    def test_contains_flexguard_products(self):
        """Registry contains FlexGuard named products."""
        from src.core.product_registry import PRODUCT_BY_NAME

        assert "FlexGuard 6Y20B" in PRODUCT_BY_NAME
        assert "FlexGuard 1Y10B" in PRODUCT_BY_NAME

    def test_maps_name_to_correct_product(self):
        """Name maps to correct product definition."""
        from src.core.product_registry import PRODUCT_BY_NAME

        product = PRODUCT_BY_NAME["FlexGuard 6Y20B"]
        assert product.code == "6Y20B"
        assert product.buffer_rate == 20


class TestGetProduct:
    """Tests for get_product function."""

    def test_lookup_by_code(self):
        """Looks up product by code."""
        from src.core.product_registry import get_product

        product = get_product("6Y20B")
        assert product.code == "6Y20B"
        assert product.product_type == "rila"

    def test_lookup_by_name(self):
        """Looks up product by display name."""
        from src.core.product_registry import get_product

        product = get_product("FlexGuard 6Y20B")
        assert product.code == "6Y20B"

    def test_lookup_by_alias(self):
        """Looks up product by alias."""
        from src.core.product_registry import get_product

        product = get_product("FG-6Y20B")
        assert product.code == "6Y20B"

    def test_raises_on_unknown_product(self):
        """Raises KeyError on unknown identifier."""
        from src.core.product_registry import get_product

        with pytest.raises(KeyError, match="Unknown product"):
            get_product("NONEXISTENT")

    def test_error_message_includes_available(self):
        """Error includes list of available products."""
        from src.core.product_registry import get_product

        with pytest.raises(KeyError, match="Available:"):
            get_product("INVALID")


class TestGetProductName:
    """Tests for get_product_name function."""

    def test_returns_display_name(self):
        """Returns human-readable display name."""
        from src.core.product_registry import get_product_name

        name = get_product_name("6Y20B")
        assert name == "FlexGuard 6Y20B"

    def test_returns_fia_name(self):
        """Returns FIA product name."""
        from src.core.product_registry import get_product_name

        name = get_product_name("FIA5YR")
        assert name == "PruSecure FIA 5-Year"


class TestGetProductType:
    """Tests for get_product_type function."""

    def test_returns_rila_type(self):
        """Returns 'rila' for RILA products."""
        from src.core.product_registry import get_product_type

        assert get_product_type("6Y20B") == "rila"
        assert get_product_type("1Y10B") == "rila"

    def test_returns_fia_type(self):
        """Returns 'fia' for FIA products."""
        from src.core.product_registry import get_product_type

        assert get_product_type("FIA5YR") == "fia"
        assert get_product_type("FIA7YR") == "fia"


class TestGetFixtureFilterName:
    """Tests for get_fixture_filter_name function."""

    def test_returns_flexguard_fixture_name(self):
        """Returns FlexGuard fixture name for RILA products."""
        from src.core.product_registry import get_fixture_filter_name

        name = get_fixture_filter_name("6Y20B")
        assert name == "FlexGuard indexed variable annuity"

    def test_returns_prusecure_fixture_name(self):
        """Returns PruSecure fixture name for FIA products."""
        from src.core.product_registry import get_fixture_filter_name

        name = get_fixture_filter_name("FIA5YR")
        assert name == "PruSecure"


class TestIsProductCode:
    """Tests for is_product_code function."""

    def test_returns_true_for_valid_code(self):
        """Returns True for known product codes."""
        from src.core.product_registry import is_product_code

        assert is_product_code("6Y20B") is True
        assert is_product_code("FIA5YR") is True
        assert is_product_code("1Y10B") is True

    def test_returns_false_for_invalid(self):
        """Returns False for unknown identifiers."""
        from src.core.product_registry import is_product_code

        assert is_product_code("INVALID") is False
        assert is_product_code("rila") is False
        assert is_product_code("") is False


class TestListProductCodes:
    """Tests for list_product_codes function."""

    def test_returns_list(self):
        """Returns a list."""
        from src.core.product_registry import list_product_codes

        codes = list_product_codes()
        assert isinstance(codes, list)

    def test_contains_rila_codes(self):
        """Contains RILA product codes."""
        from src.core.product_registry import list_product_codes

        codes = list_product_codes()
        assert "6Y20B" in codes
        assert "1Y10B" in codes

    def test_contains_fia_codes(self):
        """Contains FIA product codes."""
        from src.core.product_registry import list_product_codes

        codes = list_product_codes()
        assert "FIA5YR" in codes


class TestListProductsByType:
    """Tests for list_products_by_type function."""

    def test_lists_rila_products(self):
        """Lists all RILA products."""
        from src.core.product_registry import list_products_by_type

        rila_products = list_products_by_type("rila")
        assert len(rila_products) >= 3
        assert all(p.product_type == "rila" for p in rila_products)

    def test_lists_fia_products(self):
        """Lists all FIA products."""
        from src.core.product_registry import list_products_by_type

        fia_products = list_products_by_type("fia")
        assert len(fia_products) >= 2
        assert all(p.product_type == "fia" for p in fia_products)

    def test_returns_empty_for_myga(self):
        """Returns empty list for MYGA (none defined)."""
        from src.core.product_registry import list_products_by_type

        myga_products = list_products_by_type("myga")
        assert len(myga_products) == 0

    def test_returns_empty_for_unknown_type(self):
        """Returns empty list for unknown type."""
        from src.core.product_registry import list_products_by_type

        products = list_products_by_type("unknown")
        assert len(products) == 0


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_exports_product_definition(self):
        """Exports ProductDefinition."""
        from src.core.product_registry import __all__

        assert "ProductDefinition" in __all__

    def test_exports_lookup_functions(self):
        """Exports all lookup functions."""
        from src.core.product_registry import __all__

        expected = [
            "get_product",
            "get_product_name",
            "get_product_type",
            "get_fixture_filter_name",
            "is_product_code",
            "list_product_codes",
            "list_products_by_type",
        ]
        for func_name in expected:
            assert func_name in __all__

    def test_exports_registries(self):
        """Exports registry dictionaries."""
        from src.core.product_registry import __all__

        assert "PRODUCT_BY_CODE" in __all__
        assert "PRODUCT_BY_NAME" in __all__
