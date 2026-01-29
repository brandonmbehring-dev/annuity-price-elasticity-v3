"""
Aggregation Strategies for Competitor Rates.

Product-specific aggregation methods for competitive rate analysis.

Usage:
    from src.features.aggregation import (
        WeightedAggregation,
        TopNAggregation,
        FirmLevelAggregation,
    )
    from src.features.aggregation import get_strategy

    strategy = get_strategy("weighted")
"""

from src.features.aggregation.base import AggregationStrategyBase
from src.features.aggregation.strategies import (
    WeightedAggregation,
    TopNAggregation,
    FirmLevelAggregation,
    MedianAggregation,
)

# Import from registry to maintain single source of truth
# Pattern: same as src/products/__init__.py
from src.core.registry import AggregationRegistry


def get_strategy(strategy_name: str, **kwargs) -> AggregationStrategyBase:
    """Get an aggregation strategy by name.

    Delegates to AggregationRegistry.get() for consistency.

    Parameters
    ----------
    strategy_name : str
        Strategy name: "weighted", "top_n", "firm_level", "median"
    **kwargs
        Strategy configuration

    Returns
    -------
    AggregationStrategyBase
        Configured strategy instance
    """
    return AggregationRegistry.get(strategy_name, **kwargs)


__all__ = [
    "AggregationStrategyBase",
    "WeightedAggregation",
    "TopNAggregation",
    "FirmLevelAggregation",
    "MedianAggregation",
    "get_strategy",
]
