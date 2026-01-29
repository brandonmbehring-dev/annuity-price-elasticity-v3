"""
Product Methodologies - RILA, FIA, MYGA implementations.

Economic constraint rules and coefficient expectations by product type.

Usage:
    from src.products import RILAMethodology, FIAMethodology, MYGAMethodology
    from src.products.base import ConstraintRule

    methodology = RILAMethodology()
    rules = methodology.get_constraint_rules()

    # Or use factory function (delegates to registry):
    from src.products import get_methodology
    methodology = get_methodology("rila")
"""

from src.products.base import ProductMethodology, ConstraintRule
from src.products.rila_methodology import RILAMethodology
from src.products.fia_methodology import FIAMethodology
from src.products.myga_methodology import MYGAMethodology

# Import canonical get_methodology from registry to avoid duplication
from src.core.registry import get_methodology

# Alias for backward compatibility
get_product_methodology = get_methodology


__all__ = [
    "ProductMethodology",
    "ConstraintRule",
    "RILAMethodology",
    "FIAMethodology",
    "MYGAMethodology",
    "get_methodology",
    "get_product_methodology",  # Alias
]
