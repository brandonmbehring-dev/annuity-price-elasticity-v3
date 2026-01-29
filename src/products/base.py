"""
Product Methodology Base Protocol.

Defines the interface that all product-type specific methodology
implementations must follow.

Migrated from v1 src/products/base.py with enhancements for v2.

Usage:
    from src.products.base import ProductMethodology, ConstraintRule

    class MyProductMethodology:
        def get_constraint_rules(self) -> List[ConstraintRule]:
            ...
"""

from typing import Protocol, List, Dict

# ConstraintRule is canonical in registry.py - re-export for backward compatibility
from src.core.registry import ConstraintRule


class ProductMethodology(Protocol):
    """Protocol defining product-type specific methodology.

    All product types (RILA, FIA, MYGA) must implement this interface
    to provide their specific constraint rules and coefficient expectations.

    Methods
    -------
    get_constraint_rules()
        Return list of economic constraint rules for the product type
    get_coefficient_signs()
        Return expected coefficient signs by feature pattern
    supports_regime_detection()
        Whether the product type supports regime detection analysis
    """

    def get_constraint_rules(self) -> List[ConstraintRule]:
        """Get economic constraint rules for this product type.

        Returns
        -------
        List[ConstraintRule]
            List of constraint rules to validate model coefficients
        """
        ...

    def get_coefficient_signs(self) -> Dict[str, str]:
        """Get expected coefficient signs by feature pattern.

        Returns
        -------
        Dict[str, str]
            Mapping of feature pattern to expected sign ("positive"/"negative")
        """
        ...

    def supports_regime_detection(self) -> bool:
        """Check if product type supports regime detection.

        RILA and FIA use yield-based rules (same methodology).
        MYGA may need regime detection for rate environment changes.

        Returns
        -------
        bool
            True if regime detection is appropriate for this product type
        """
        ...

    @property
    def product_type(self) -> str:
        """Return the product type identifier.

        Returns
        -------
        str
            Product type: "rila", "fia", or "myga"
        """
        ...


__all__ = ["ProductMethodology", "ConstraintRule"]
