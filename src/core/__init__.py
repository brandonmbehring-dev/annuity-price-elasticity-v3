"""
Core Module - Protocols, Types, Registries, and Exceptions.

Central abstractions for the annuity price elasticity system.

Usage:
    from src.core import protocols, types, registry, exceptions
    from src.core.protocols import DataSourceAdapter
    from src.core.registry import get_adapter, get_methodology
    from src.core.exceptions import DataLoadError, ConstraintViolationError
"""

from src.core.protocols import (
    DataSourceAdapter,
    AggregationStrategy,
    ProductMethodology,
    NotebookInterfaceProtocol,
)
from src.core.types import (
    AWSConfig,
    InferenceConfig,
    FeatureConfig,
    DataPaths,
    AggregationConfig,
    ConstraintConfig,
    InferenceResults,
)
# FeatureSelectionResults moved to selection_types for DRY compliance
from src.features.selection_types import FeatureSelectionResults

# Exception hierarchy for fail-fast error handling
from src.core.exceptions import (
    ElasticityBaseError,
    DataError,
    DataLoadError,
    DataValidationError,
    DataSchemaError,
    DiagnosticError,
    AutocorrelationTestError,
    HeteroscedasticityTestError,
    NormalityTestError,
    MulticollinearityError,
    ModelError,
    ConstraintViolationError,
    ModelConvergenceError,
    FeatureSelectionError,
    VisualizationError,
    PlotGenerationError,
    ExportError,
    ConfigurationError,
    InvalidConfigError,
    ProductNotFoundError,
)

__all__ = [
    # Protocols
    "DataSourceAdapter",
    "AggregationStrategy",
    "ProductMethodology",
    "NotebookInterfaceProtocol",
    # Types
    "AWSConfig",
    "InferenceConfig",
    "FeatureConfig",
    "DataPaths",
    "AggregationConfig",
    "ConstraintConfig",
    "InferenceResults",
    "FeatureSelectionResults",
    # Exceptions
    "ElasticityBaseError",
    "DataError",
    "DataLoadError",
    "DataValidationError",
    "DataSchemaError",
    "DiagnosticError",
    "AutocorrelationTestError",
    "HeteroscedasticityTestError",
    "NormalityTestError",
    "MulticollinearityError",
    "ModelError",
    "ConstraintViolationError",
    "ModelConvergenceError",
    "FeatureSelectionError",
    "VisualizationError",
    "PlotGenerationError",
    "ExportError",
    "ConfigurationError",
    "InvalidConfigError",
    "ProductNotFoundError",
]
