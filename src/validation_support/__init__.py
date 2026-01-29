"""
Validation Support Module

Provides validation utilities for offline development and mathematical equivalence validation.
(Renamed from src/testing/ to avoid pytest confusion)

Modules:
    - aws_mock_layer: Mock AWS operations for offline notebook development
    - mathematical_equivalence: Validate mathematical equivalence during refactoring (consolidated)

Mathematical Equivalence (Phase 11 Consolidation):
    This module now exports all mathematical equivalence validation functionality
    from a single consolidated source. Prior modules have been removed:
    - src/data/mathematical_equivalence_validator.py (deleted)
    - src/features/selection/mathematical_equivalence_validator.py (deleted)

    All imports should now use:
        from src.validation_support.mathematical_equivalence import ...
"""

from .aws_mock_layer import (
    setup_offline_environment,
    verify_fixture_availability,
    is_offline_mode,
    OfflineS3Resource,
    OfflineSTSClient
)

from .mathematical_equivalence import (
    # Constants
    TOLERANCE,
    BOOTSTRAP_STATISTICAL_TOLERANCE,
    MLFLOW_AVAILABLE,
    # Dataclasses
    ValidationResult,
    EquivalenceValidationResult,
    EquivalenceResult,
    # Classes
    MathematicalEquivalenceValidator,
    DataFrameEquivalenceValidator,
    # Exception
    MathematicalEquivalenceError,
    # Convenience functions
    validate_mathematical_equivalence_comprehensive,
    validate_pipeline_stage_equivalence,
    enforce_transformation_equivalence,
    validate_baseline_equivalence,
)

__all__ = [
    # AWS mock layer
    'setup_offline_environment',
    'verify_fixture_availability',
    'is_offline_mode',
    'OfflineS3Resource',
    'OfflineSTSClient',
    # Mathematical equivalence - Constants
    'TOLERANCE',
    'BOOTSTRAP_STATISTICAL_TOLERANCE',
    'MLFLOW_AVAILABLE',
    # Mathematical equivalence - Dataclasses
    'ValidationResult',
    'EquivalenceValidationResult',
    'EquivalenceResult',
    # Mathematical equivalence - Classes
    'MathematicalEquivalenceValidator',
    'DataFrameEquivalenceValidator',
    # Mathematical equivalence - Exception
    'MathematicalEquivalenceError',
    # Mathematical equivalence - Functions
    'validate_mathematical_equivalence_comprehensive',
    'validate_pipeline_stage_equivalence',
    'enforce_transformation_equivalence',
    'validate_baseline_equivalence',
]
