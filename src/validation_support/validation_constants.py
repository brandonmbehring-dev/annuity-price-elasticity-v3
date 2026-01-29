"""
Shared Constants and Types for Mathematical Equivalence Validation.

This module contains shared constants, dataclasses, and exceptions used
by all validation modules. Extracted for DRY compliance.

Module Architecture (Phase 6.3 Split):
- validation_constants.py: Shared constants, dataclasses, exception (this file)
- validation_feature_selection.py: MathematicalEquivalenceValidator + related
- validation_dataframe.py: DataFrameEquivalenceValidator + related
- mathematical_equivalence.py: Thin wrapper with re-exports
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# =============================================================================
# SHARED CONSTANTS
# =============================================================================

# Default numerical tolerance (CODING_STANDARDS.md §4)
TOLERANCE = 1e-12

# Bootstrap comparison tolerance (stochastic processes have higher variance)
BOOTSTRAP_STATISTICAL_TOLERANCE = 1e-6


# =============================================================================
# DATACLASSES - Validation Results
# =============================================================================


@dataclass
class ValidationResult:
    """Container for mathematical equivalence validation results (feature selection)."""

    test_name: str
    passed: bool
    max_difference: float
    tolerance: float
    baseline_value: Any
    test_value: Any
    details: Dict[str, Any]
    timestamp: str


@dataclass
class EquivalenceValidationResult:
    """Container for DataFrame transformation equivalence validation results."""

    # Validation Metadata
    transformation_name: str
    timestamp: str
    tolerance: float

    # Shape Validation
    shapes_match: bool
    original_shape: Tuple[int, int]
    transformed_shape: Tuple[int, int]

    # Content Validation
    values_equivalent: bool
    max_absolute_difference: float
    max_relative_difference: float
    columns_compared: int

    # Detailed Comparison
    equivalent_columns: List[str]
    non_equivalent_columns: List[Dict[str, Any]]

    # Overall Result
    mathematically_equivalent: bool
    validation_passed: bool

    # Business Context
    business_impact_assessment: str
    recommendation: str


@dataclass
class EquivalenceResult:
    """Container for comprehensive mathematical equivalence validation result."""

    comparison_type: str
    validation_passed: bool
    tolerance_used: float
    differences_found: List[Dict[str, Any]]
    max_absolute_difference: float
    max_relative_difference: float
    summary_metrics: Dict[str, float]
    business_interpretation: str
    remediation_required: bool
    remediation_suggestions: List[str]


# =============================================================================
# EXCEPTIONS
# =============================================================================


class MathematicalEquivalenceError(Exception):
    """Exception raised when mathematical equivalence validation fails."""
    pass
