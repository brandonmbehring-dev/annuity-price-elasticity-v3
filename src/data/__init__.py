"""
Data processing modules for Annuity Price Elasticity v2.

Exports:
- extraction: AWS/S3 data loading operations
- preprocessing: Data cleaning and validation
- pipelines: Data pipeline orchestration
- quality_monitor: Data quality scoring
- adapters: DI-compatible data source adapters

Note: To avoid circular imports, use submodule imports directly:
    from src.data.adapters import get_adapter
    from src.data.extraction import create_sts_client
"""

__all__ = [
    "adapters",
    "extraction",
    "preprocessing",
    "pipelines",
]
