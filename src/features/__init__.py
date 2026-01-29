"""
Feature engineering modules for Annuity Price Elasticity v2.

Exports:
- aggregation: Competitor rate aggregation strategies
- selection: Feature selection engines and interface

Note: To avoid circular imports, use submodule imports directly:
    from src.features.aggregation import get_strategy
    from src.features.selection import FeatureSelectionEngine
"""

__all__ = [
    "aggregation",
    "selection",
]
