"""
Data Adapters - S3, Local, and Fixture implementations.

Dependency injection pattern for data source abstraction.

Usage:
    from src.data.adapters import S3Adapter, LocalAdapter, FixtureAdapter
    from src.data.adapters import get_adapter

    # Direct instantiation
    adapter = S3Adapter(config)

    # Via factory function (delegates to registry)
    adapter = get_adapter("aws", config=aws_config)
"""

from src.data.adapters.base import DataAdapterBase
from src.data.adapters.s3_adapter import S3Adapter
from src.data.adapters.local_adapter import LocalAdapter
from src.data.adapters.fixture_adapter import FixtureAdapter

# Import canonical get_adapter from registry to avoid duplication
# Pattern: same as src/products/__init__.py
from src.core.registry import get_adapter


__all__ = [
    "DataAdapterBase",
    "S3Adapter",
    "LocalAdapter",
    "FixtureAdapter",
    "get_adapter",
]
