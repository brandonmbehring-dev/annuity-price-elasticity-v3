"""
Multiple Testing Correction Subpackage.

Provides FWER/FDR corrections for feature selection:
- Bonferroni correction (FWER control)
- FDR correction (Benjamini-Hochberg)
- Search space reduction strategies
"""

from src.features.selection.enhancements.multiple_testing.multiple_testing_correction import (
    compare_correction_methods,
)
from src.features.selection.enhancements.multiple_testing.bonferroni_engine import (
    apply_bonferroni_correction,
)
from src.features.selection.enhancements.multiple_testing.fdr_engine import (
    apply_fdr_correction,
)
from src.features.selection.enhancements.multiple_testing.search_space_reduction import (
    create_reduced_search_space,
)
from src.features.selection.enhancements.multiple_testing.multiple_testing_types import (
    MultipleTestingResults,
)

# Alias for backward compatibility and cleaner API
apply_multiple_testing_correction = compare_correction_methods
reduce_search_space = create_reduced_search_space

__all__ = [
    # Primary API
    "compare_correction_methods",
    "apply_multiple_testing_correction",  # Alias
    "create_reduced_search_space",
    "reduce_search_space",  # Alias
    # Component functions
    "apply_bonferroni_correction",
    "apply_fdr_correction",
    # Types
    "MultipleTestingResults",
]
