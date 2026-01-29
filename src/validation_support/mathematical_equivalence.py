"""
Consolidated Mathematical Equivalence Validation Framework.

This module provides comprehensive validation of mathematical calculations
to ensure refactoring and transformations preserve numerical precision.

Key Validation Domains:
1. Feature Selection: AIC calculations, bootstrap stability, economic constraints
2. Data Pipelines: DataFrame transformations, data preprocessing
3. Model Training: Coefficient comparisons, inference validation

Design Principles:
- Single source of truth for equivalence validation (consolidates 3 prior modules)
- 1e-12 precision tolerance for mathematical calculations (target: 0.00e+00)
- Fail-fast behavior with comprehensive business context
- MLflow integration for validation tracking
- Support for both regression testing and enhancement validation

Mathematical Foundation:
- AIC = 2k - 2 * ln(L) where k=parameters, L=max likelihood
- Bootstrap stability CV = std(AIC_bootstrap) / mean(AIC_bootstrap)
- Economic constraints: competitor coefficients < 0, prudential > 0

Usage:
    # Feature Selection Validation
    from src.validation_support.mathematical_equivalence import (
        MathematicalEquivalenceValidator,
        validate_mathematical_equivalence_comprehensive
    )

    # DataFrame Transformation Validation
    from src.validation_support.mathematical_equivalence import (
        DataFrameEquivalenceValidator,
        validate_pipeline_stage_equivalence,
        enforce_transformation_equivalence
    )

Module Architecture (Phase 6.3 Split):
- validation_constants.py: Shared constants, dataclasses, exception
- validation_feature_selection.py: MathematicalEquivalenceValidator + related
- validation_dataframe.py: DataFrameEquivalenceValidator + related
- mathematical_equivalence.py: Thin wrapper with re-exports (this file)

Migration Note (Phase 11 + Rename):
    This module consolidates:
    - src/validation_support/mathematical_equivalence.py (renamed from src/testing/)
    - src/data/mathematical_equivalence_validator.py (deleted)
    - src/features/selection/mathematical_equivalence_validator.py (deleted)

    All prior imports should now use src.validation_support.
"""

import logging

import numpy as np

# Configure for high precision mathematical operations
np.set_printoptions(precision=15)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES
# =============================================================================

# Shared constants and types (validation_constants.py)
from src.validation_support.validation_constants import (
    TOLERANCE,
    BOOTSTRAP_STATISTICAL_TOLERANCE,
    ValidationResult,
    EquivalenceValidationResult,
    EquivalenceResult,
    MathematicalEquivalenceError,
)

# Feature selection validation (validation_feature_selection.py)
from src.validation_support.validation_feature_selection import (
    MathematicalEquivalenceValidator,
    validate_mathematical_equivalence_comprehensive,
    _compare_models_for_equivalence,
)

# DataFrame transformation validation (validation_dataframe.py)
from src.validation_support.validation_dataframe import (
    DataFrameEquivalenceValidator,
    validate_pipeline_stage_equivalence,
    enforce_transformation_equivalence,
    validate_baseline_equivalence,
    _compare_dataframes_for_equivalence,
    _generate_suggestions,
    MLFLOW_AVAILABLE,
)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'TOLERANCE',
    'BOOTSTRAP_STATISTICAL_TOLERANCE',
    'MLFLOW_AVAILABLE',

    # Dataclasses
    'ValidationResult',
    'EquivalenceValidationResult',
    'EquivalenceResult',

    # Exceptions
    'MathematicalEquivalenceError',

    # Feature Selection Validation
    'MathematicalEquivalenceValidator',
    'validate_mathematical_equivalence_comprehensive',

    # DataFrame Transformation Validation
    'DataFrameEquivalenceValidator',
    'validate_pipeline_stage_equivalence',
    'enforce_transformation_equivalence',
    'validate_baseline_equivalence',

    # Internal helpers (for backward compatibility)
    '_compare_models_for_equivalence',
    '_compare_dataframes_for_equivalence',
    '_generate_suggestions',
]
