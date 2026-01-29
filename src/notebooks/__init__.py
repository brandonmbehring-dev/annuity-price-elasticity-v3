"""
Notebook Interface Module.

Unified API for multi-product price elasticity analysis.

Usage:
    from src.notebooks import UnifiedNotebookInterface, create_interface

    # AWS production
    interface = create_interface("6Y20B", environment="aws")

    # Testing
    interface = create_interface("6Y20B", environment="test")
"""

from src.notebooks.interface import UnifiedNotebookInterface, create_interface

__all__ = [
    "UnifiedNotebookInterface",
    "create_interface",
]
