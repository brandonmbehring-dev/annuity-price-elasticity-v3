"""
Shared Types for Statistical Constraints Engine.

This module contains shared data structures used by all statistical
constraint validation modules.

Used by: statistical_validators.py, constraint_analyzers.py,
         statistical_constraints_engine.py
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple


class ConstraintType(Enum):
    """Economic constraint types for business theory validation."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MAGNITUDE = "magnitude"
    RATIO = "ratio"


@dataclass
class StatisticalConstraintResult:
    """
    Container for statistical constraint validation result.

    Attributes
    ----------
    feature_name : str
        Feature being validated
    constraint_type : ConstraintType
        Type of economic constraint
    coefficient_estimate : float
        Point estimate of coefficient
    standard_error : float
        Standard error of coefficient estimate
    confidence_interval : Tuple[float, float]
        Confidence interval (e.g., 95%)
    t_statistic : float
        t-statistic for significance test
    p_value : float
        p-value for coefficient significance
    statistically_significant : bool
        Whether coefficient is significantly different from zero
    constraint_satisfied : bool
        Whether statistical constraint is satisfied
    constraint_strength : str
        Strength of constraint satisfaction ('STRONG', 'MODERATE', 'WEAK', 'VIOLATED')
    business_interpretation : str
        Business-friendly interpretation
    hard_threshold_comparison : Dict[str, bool]
        Comparison with hard threshold approach
    """
    feature_name: str
    constraint_type: ConstraintType
    coefficient_estimate: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    t_statistic: float
    p_value: float
    statistically_significant: bool
    constraint_satisfied: bool
    constraint_strength: str
    business_interpretation: str
    hard_threshold_comparison: Dict[str, bool]


@dataclass
class ComprehensiveConstraintAnalysis:
    """
    Container for complete statistical constraint analysis.

    Attributes
    ----------
    model_specification : Dict[str, Any]
        Model details and sample information
    constraint_results : List[StatisticalConstraintResult]
        Individual constraint validation results
    overall_assessment : Dict[str, Any]
        Summary assessment across all constraints
    methodology_comparison : Dict[str, Any]
        Hard threshold vs statistical method comparison
    business_recommendations : Dict[str, str]
        Actionable business recommendations
    power_analysis : Dict[str, float]
        Statistical power assessment for constraint detection
    """
    model_specification: Dict[str, Any]
    constraint_results: List[StatisticalConstraintResult]
    overall_assessment: Dict[str, Any]
    methodology_comparison: Dict[str, Any]
    business_recommendations: Dict[str, str]
    power_analysis: Dict[str, float]


__all__ = [
    'ConstraintType',
    'StatisticalConstraintResult',
    'ComprehensiveConstraintAnalysis',
]
