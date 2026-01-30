"""
Product Registry - Single source of truth for product definitions.

Provides unified lookup by code, name, or alias across the codebase.
Maps product codes (6Y20B) to:
- Product names in fixtures ("FlexGuard indexed variable annuity")
- Product types for methodology lookup ("rila", "fia", "myga")

Usage:
    from src.core.product_registry import (
        get_product,
        get_product_name,
        get_product_type,
        get_fixture_filter_name,
        PRODUCT_BY_CODE,
    )

    # Get full product definition
    product = get_product("6Y20B")

    # Get product type for methodology
    product_type = get_product_type("6Y20B")  # Returns "rila"

    # Get fixture filter name
    filter_name = get_fixture_filter_name("6Y20B")
    # Returns "FlexGuard indexed variable annuity"
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ProductDefinition:
    """Immutable product definition.

    Central definition for product metadata used throughout the codebase.
    Provides mapping between different product identifiers.

    Attributes
    ----------
    code : str
        Short product code used in configs and exports (e.g., "6Y20B")
    name : str
        Human-readable display name (e.g., "FlexGuard 6Y20B")
    product_type : str
        Product category: "rila", "fia", or "myga"
    fixture_name : str
        Product name as it appears in fixture/sales data
        (e.g., "FlexGuard indexed variable annuity")
    buffer_rate : Optional[int]
        Buffer percentage (e.g., 20). None for FIA/MYGA.
    term_years : int
        Product term in years
    aliases : Tuple[str, ...]
        Alternative names for lookup
    """

    code: str
    name: str
    product_type: str
    fixture_name: str
    buffer_rate: Optional[int]
    term_years: int
    aliases: Tuple[str, ...] = field(default_factory=tuple)


# =============================================================================
# PRODUCT DEFINITIONS - Single Source of Truth
# =============================================================================

_PRODUCTS: Tuple[ProductDefinition, ...] = (
    # RILA Products
    ProductDefinition(
        code="6Y20B",
        name="FlexGuard 6Y20B",
        product_type="rila",
        fixture_name="FlexGuard indexed variable annuity",
        buffer_rate=20,
        term_years=6,
        aliases=("FlexGuard 6Y20B", "FG-6Y20B", "RILA-6Y20B"),
    ),
    ProductDefinition(
        code="6Y10B",
        name="FlexGuard 6Y10B",
        product_type="rila",
        fixture_name="FlexGuard indexed variable annuity",
        buffer_rate=10,
        term_years=6,
        aliases=("FlexGuard 6Y10B", "FG-6Y10B", "RILA-6Y10B"),
    ),
    ProductDefinition(
        code="10Y20B",
        name="FlexGuard 10Y20B",
        product_type="rila",
        fixture_name="FlexGuard indexed variable annuity",
        buffer_rate=20,
        term_years=10,
        aliases=("FlexGuard 10Y20B", "FG-10Y20B", "RILA-10Y20B"),
    ),
    ProductDefinition(
        code="1Y10B",
        name="FlexGuard 1Y10B",
        product_type="rila",
        fixture_name="FlexGuard indexed variable annuity",
        buffer_rate=10,
        term_years=1,
        aliases=("FlexGuard 1Y10B", "FG-1Y10B", "RILA-1Y10B"),
    ),
    # FIA Products
    ProductDefinition(
        code="FIA5YR",
        name="PruSecure FIA 5-Year",
        product_type="fia",
        fixture_name="PruSecure",
        buffer_rate=None,
        term_years=5,
        aliases=("FIA-5YR", "PruSecure 5Y"),
    ),
    ProductDefinition(
        code="FIA7YR",
        name="PruSecure FIA 7-Year",
        product_type="fia",
        fixture_name="PruSecure",
        buffer_rate=None,
        term_years=7,
        aliases=("FIA-7YR", "PruSecure 7Y"),
    ),
    ProductDefinition(
        code="FIACA5YR",
        name="PruSecure FIA Cap 5-Year",
        product_type="fia",
        fixture_name="PruSecure",
        buffer_rate=None,
        term_years=5,
        aliases=("FIA-CAP-5YR",),
    ),
    ProductDefinition(
        code="FIACA7YR",
        name="PruSecure FIA Cap 7-Year",
        product_type="fia",
        fixture_name="PruSecure",
        buffer_rate=None,
        term_years=7,
        aliases=("FIA-CAP-7YR",),
    ),
)

# =============================================================================
# LOOKUP INDICES
# =============================================================================

PRODUCT_BY_CODE: Dict[str, ProductDefinition] = {p.code: p for p in _PRODUCTS}
PRODUCT_BY_NAME: Dict[str, ProductDefinition] = {p.name: p for p in _PRODUCTS}

# Build alias lookup (maps aliases to their product)
_ALIAS_LOOKUP: Dict[str, ProductDefinition] = {}
for product in _PRODUCTS:
    for alias in product.aliases:
        _ALIAS_LOOKUP[alias] = product


# =============================================================================
# LOOKUP FUNCTIONS
# =============================================================================


def get_product(identifier: str) -> ProductDefinition:
    """Lookup product by code, name, or alias.

    Parameters
    ----------
    identifier : str
        Product code (e.g., "6Y20B"), name, or alias

    Returns
    -------
    ProductDefinition
        Full product definition

    Raises
    ------
    KeyError
        If product not found

    Examples
    --------
    >>> product = get_product("6Y20B")
    >>> product.product_type
    'rila'
    >>> product.fixture_name
    'FlexGuard indexed variable annuity'
    """
    # Try code first (most common)
    if identifier in PRODUCT_BY_CODE:
        return PRODUCT_BY_CODE[identifier]

    # Try name
    if identifier in PRODUCT_BY_NAME:
        return PRODUCT_BY_NAME[identifier]

    # Try aliases
    if identifier in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[identifier]

    available = list(PRODUCT_BY_CODE.keys())
    raise KeyError(f"Unknown product: '{identifier}'. Available: {available}")


def get_product_name(code: str) -> str:
    """Get product display name from code.

    Parameters
    ----------
    code : str
        Product code (e.g., "6Y20B")

    Returns
    -------
    str
        Human-readable product name

    Examples
    --------
    >>> get_product_name("6Y20B")
    'FlexGuard 6Y20B'
    """
    return get_product(code).name


def get_product_type(code: str) -> str:
    """Get product type from code.

    Parameters
    ----------
    code : str
        Product code (e.g., "6Y20B")

    Returns
    -------
    str
        Product type: "rila", "fia", or "myga"

    Examples
    --------
    >>> get_product_type("6Y20B")
    'rila'
    >>> get_product_type("FIA5YR")
    'fia'
    """
    return get_product(code).product_type


def get_fixture_filter_name(code: str) -> str:
    """Get the product name as it appears in fixture data.

    This is the value to filter by in the 'product_name' column
    of sales fixture data.

    Parameters
    ----------
    code : str
        Product code (e.g., "6Y20B")

    Returns
    -------
    str
        Fixture product name for filtering

    Examples
    --------
    >>> get_fixture_filter_name("6Y20B")
    'FlexGuard indexed variable annuity'
    """
    return get_product(code).fixture_name


def is_product_code(identifier: str) -> bool:
    """Check if identifier is a known product code.

    Parameters
    ----------
    identifier : str
        String to check

    Returns
    -------
    bool
        True if identifier is a product code

    Examples
    --------
    >>> is_product_code("6Y20B")
    True
    >>> is_product_code("rila")
    False
    """
    return identifier in PRODUCT_BY_CODE


def list_product_codes() -> list:
    """List all available product codes.

    Returns
    -------
    List[str]
        All registered product codes

    Examples
    --------
    >>> codes = list_product_codes()
    >>> "6Y20B" in codes
    True
    """
    return list(PRODUCT_BY_CODE.keys())


def list_products_by_type(product_type: str) -> list:
    """List all products of a given type.

    Parameters
    ----------
    product_type : str
        Product type: "rila", "fia", or "myga"

    Returns
    -------
    List[ProductDefinition]
        Products matching the type

    Examples
    --------
    >>> rila_products = list_products_by_type("rila")
    >>> len(rila_products) >= 3
    True
    """
    return [p for p in _PRODUCTS if p.product_type == product_type]


__all__ = [
    "ProductDefinition",
    "PRODUCT_BY_CODE",
    "PRODUCT_BY_NAME",
    "get_product",
    "get_product_name",
    "get_product_type",
    "get_fixture_filter_name",
    "is_product_code",
    "list_product_codes",
    "list_products_by_type",
]
