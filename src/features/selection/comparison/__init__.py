"""
Methodology Comparison Subpackage.

Provides comparison tools for analyzing different feature selection approaches:
- comparative_analysis: Cross-method comparison framework
- comparison_metrics: Statistical comparison metrics
- comparison_business: Business impact analysis
"""

from src.features.selection.comparison.comparative_analysis import (
    run_comparative_analysis,
)
from src.features.selection.comparison.comparison_metrics import (
    calculate_comparison_metrics,
)
from src.features.selection.comparison.comparison_business import (
    analyze_business_impact,
)

__all__ = [
    "run_comparative_analysis",
    "calculate_comparison_metrics",
    "analyze_business_impact",
]
