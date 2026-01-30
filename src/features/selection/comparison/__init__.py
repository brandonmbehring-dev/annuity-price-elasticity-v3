"""
Methodology Comparison Subpackage.

Provides comparison tools for analyzing different feature selection approaches:
- comparative_analysis: Cross-method comparison framework
- comparison_metrics: Statistical comparison metrics
- comparison_business: Business impact analysis
"""

from src.features.selection.comparison.comparative_analysis import (
    compare_methodologies,
)
from src.features.selection.comparison.comparison_business import (
    _analyze_business_impact,
)

__all__ = [
    "compare_methodologies",
    "_analyze_business_impact",
]
