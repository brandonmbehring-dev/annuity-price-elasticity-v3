"""
Statistical Constraints Subpackage.

Provides CI-based constraint validation for feature selection:
- Statistical significance-based filtering
- Confidence interval constraints
- Advanced constraint analysis
"""

from src.features.selection.enhancements.statistical_constraints.statistical_constraints_engine import (
    apply_statistical_constraints,
)
from src.features.selection.enhancements.statistical_constraints.constraint_types import (
    ConstraintType,
    StatisticalConstraintResult,
    ComprehensiveConstraintAnalysis,
)

__all__ = [
    # Primary API
    "apply_statistical_constraints",
    # Types
    "ConstraintType",
    "StatisticalConstraintResult",
    "ComprehensiveConstraintAnalysis",
]
