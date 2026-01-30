"""
Registry System for Annuity Price Elasticity v2.

Centralized registration of product methodologies, aggregation strategies,
and data adapters. Supports runtime discovery and configuration.

Usage:
    from src.core.registry import (
        BusinessRulesRegistry,
        AdapterRegistry,
        get_methodology,
        get_adapter,
    )

    # Get methodology for a product type
    methodology = get_methodology("rila")

    # Get adapter by type
    adapter = get_adapter("aws", config=aws_config)
"""

from typing import Dict, Type, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from src.core.protocols import (
    ProductMethodology,
    DataSourceAdapter,
    AggregationStrategy,
)


# =============================================================================
# BASE REGISTRY (DRY Pattern)
# =============================================================================


class BaseRegistry(ABC):
    """Abstract base class for registries with lazy initialization.

    Provides common initialization pattern to eliminate duplication
    across BusinessRulesRegistry, AdapterRegistry, and AggregationRegistry.

    Subclasses must implement:
        - _register_defaults(): Register default implementations
        - _initialized: Class-level bool flag
    """

    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Initialize default registrations if not already done.

        This method is idempotent and safe to call multiple times.
        """
        if not cls._initialized:
            cls._register_defaults()
            cls._initialized = True

    @classmethod
    @abstractmethod
    def _register_defaults(cls) -> None:
        """Register default implementations. Must be overridden by subclasses."""
        pass


# =============================================================================
# CONSTRAINT RULE DATACLASS
# =============================================================================


@dataclass(frozen=True)
class ConstraintRule:
    """Single economic constraint rule definition.

    Adapted from src/products/base.py in v1.

    Attributes
    ----------
    feature_pattern : str
        Pattern to match feature names (e.g., "competitor_")
    expected_sign : str
        Expected coefficient sign: "positive" or "negative"
    constraint_type : str
        Type identifier (e.g., "COMPETITOR_NEGATIVE")
    business_rationale : str
        Explanation for the constraint
    strict : bool
        Whether violation should fail (True) or warn (False)
    """

    feature_pattern: str
    expected_sign: str
    constraint_type: str
    business_rationale: str
    strict: bool = True


# =============================================================================
# BUSINESS RULES REGISTRY
# =============================================================================


class BusinessRulesRegistry(BaseRegistry):
    """Registry for product-specific business rules and methodologies.

    Singleton pattern ensures consistent rule access across modules.
    Inherits lazy initialization from BaseRegistry.

    Class Methods
    -------------
    register(product_type, methodology)
        Register a methodology implementation
    get(product_type)
        Get registered methodology
    get_constraint_rules(product_type)
        Get constraint rules for a product type
    list_registered()
        List all registered product types
    """

    _methodologies: Dict[str, ProductMethodology] = {}
    _initialized: bool = False  # Override for class-specific state

    @classmethod
    def register(
        cls, product_type: str, methodology: ProductMethodology
    ) -> None:
        """Register a methodology implementation.

        Parameters
        ----------
        product_type : str
            Product type identifier: "rila", "fia", "myga"
        methodology : ProductMethodology
            Methodology implementation

        Raises
        ------
        ValueError
            If product_type is invalid
        """
        valid_types = {"rila", "fia", "myga"}
        if product_type not in valid_types:
            raise ValueError(
                f"Invalid product_type: {product_type}. "
                f"Must be one of: {valid_types}"
            )
        cls._methodologies[product_type] = methodology

    @classmethod
    def get(cls, product_type: str) -> ProductMethodology:
        """Get registered methodology for a product type.

        Parameters
        ----------
        product_type : str
            Product type identifier

        Returns
        -------
        ProductMethodology
            Registered methodology

        Raises
        ------
        KeyError
            If product_type is not registered
        """
        cls._ensure_initialized()
        if product_type not in cls._methodologies:
            available = ", ".join(cls._methodologies.keys())
            raise KeyError(
                f"No methodology registered for '{product_type}'. "
                f"Available: {available}"
            )
        return cls._methodologies[product_type]

    @classmethod
    def get_constraint_rules(cls, product_type: str) -> list:
        """Get constraint rules for a product type.

        Parameters
        ----------
        product_type : str
            Product type identifier

        Returns
        -------
        List[ConstraintRule]
            List of constraint rules
        """
        methodology = cls.get(product_type)
        return methodology.get_constraint_rules()

    @classmethod
    def list_registered(cls) -> list:
        """List all registered product types.

        Returns
        -------
        List[str]
            Registered product types
        """
        cls._ensure_initialized()
        return list(cls._methodologies.keys())

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default methodology implementations."""
        # Import here to avoid circular imports
        from src.products.rila_methodology import RILAMethodology
        from src.products.fia_methodology import FIAMethodology
        from src.products.myga_methodology import MYGAMethodology

        cls.register("rila", RILAMethodology())
        cls.register("fia", FIAMethodology())
        cls.register("myga", MYGAMethodology())


# =============================================================================
# ADAPTER REGISTRY
# =============================================================================


class AdapterRegistry(BaseRegistry):
    """Registry for data source adapters.

    Manages adapter factory functions for different data sources.
    Inherits lazy initialization from BaseRegistry.

    Class Methods
    -------------
    register(adapter_type, factory)
        Register an adapter factory
    get(adapter_type, **kwargs)
        Get/create an adapter instance
    list_registered()
        List available adapter types
    """

    _factories: Dict[str, Callable[..., DataSourceAdapter]] = {}
    _initialized: bool = False  # Override for class-specific state

    @classmethod
    def register(
        cls,
        adapter_type: str,
        factory: Callable[..., DataSourceAdapter],
    ) -> None:
        """Register an adapter factory function.

        Parameters
        ----------
        adapter_type : str
            Adapter type: "aws", "local", "fixture"
        factory : Callable
            Factory function that creates adapter instances
        """
        cls._factories[adapter_type] = factory

    @classmethod
    def get(cls, adapter_type: str, **kwargs) -> DataSourceAdapter:
        """Get/create an adapter instance.

        Parameters
        ----------
        adapter_type : str
            Adapter type to create
        **kwargs
            Arguments passed to the factory function

        Returns
        -------
        DataSourceAdapter
            Configured adapter instance

        Raises
        ------
        KeyError
            If adapter_type is not registered
        """
        cls._ensure_initialized()
        if adapter_type not in cls._factories:
            available = ", ".join(cls._factories.keys())
            raise KeyError(
                f"No adapter registered for '{adapter_type}'. "
                f"Available: {available}"
            )
        return cls._factories[adapter_type](**kwargs)

    @classmethod
    def list_registered(cls) -> list:
        """List available adapter types.

        Returns
        -------
        List[str]
            Registered adapter types
        """
        cls._ensure_initialized()
        return list(cls._factories.keys())

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default adapter factories."""
        # Import here to avoid circular imports
        from src.data.adapters.s3_adapter import S3Adapter
        from src.data.adapters.local_adapter import LocalAdapter
        from src.data.adapters.fixture_adapter import FixtureAdapter

        cls.register("aws", lambda config: S3Adapter(config))
        cls.register("local", lambda data_dir: LocalAdapter(data_dir))
        cls.register("fixture", lambda fixtures_dir: FixtureAdapter(fixtures_dir))


# =============================================================================
# AGGREGATION STRATEGY REGISTRY
# =============================================================================


class AggregationRegistry(BaseRegistry):
    """Registry for competitor aggregation strategies.

    Inherits lazy initialization from BaseRegistry.

    Class Methods
    -------------
    register(strategy_name, factory)
        Register a strategy factory
    get(strategy_name, **kwargs)
        Get/create a strategy instance
    get_for_product_type(product_type)
        Get default strategy for a product type
    """

    _factories: Dict[str, Callable[..., AggregationStrategy]] = {}
    _product_defaults: Dict[str, str] = {
        "rila": "weighted",
        "fia": "top_n",
        "myga": "firm_level",
    }
    _initialized: bool = False  # Override for class-specific state

    @classmethod
    def register(
        cls,
        strategy_name: str,
        factory: Callable[..., AggregationStrategy],
    ) -> None:
        """Register an aggregation strategy factory.

        Parameters
        ----------
        strategy_name : str
            Strategy name: "weighted", "top_n", "firm_level"
        factory : Callable
            Factory function that creates strategy instances
        """
        cls._factories[strategy_name] = factory

    @classmethod
    def get(cls, strategy_name: str, **kwargs) -> AggregationStrategy:
        """Get/create a strategy instance.

        Parameters
        ----------
        strategy_name : str
            Strategy to create
        **kwargs
            Arguments passed to the factory

        Returns
        -------
        AggregationStrategy
            Configured strategy instance
        """
        cls._ensure_initialized()
        if strategy_name not in cls._factories:
            available = ", ".join(cls._factories.keys())
            raise KeyError(
                f"No strategy registered for '{strategy_name}'. "
                f"Available: {available}"
            )
        return cls._factories[strategy_name](**kwargs)

    @classmethod
    def get_for_product_type(
        cls, product_type: str, **kwargs
    ) -> AggregationStrategy:
        """Get default aggregation strategy for a product type.

        Parameters
        ----------
        product_type : str
            Product type: "rila", "fia", "myga"
        **kwargs
            Arguments passed to the factory

        Returns
        -------
        AggregationStrategy
            Default strategy for the product type
        """
        if product_type not in cls._product_defaults:
            raise KeyError(f"No default strategy for product type: {product_type}")
        strategy_name = cls._product_defaults[product_type]
        return cls.get(strategy_name, **kwargs)

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default strategy factories."""
        from src.features.aggregation.strategies import (
            WeightedAggregation,
            TopNAggregation,
            FirmLevelAggregation,
            MedianAggregation,
        )

        cls.register("weighted", lambda **kw: WeightedAggregation(**kw))
        cls.register("top_n", lambda **kw: TopNAggregation(**kw))
        cls.register("firm_level", lambda **kw: FirmLevelAggregation(**kw))
        cls.register("median", lambda **kw: MedianAggregation(**kw))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_methodology(product_identifier: str) -> ProductMethodology:
    """Get methodology for a product code or type.

    Accepts either a product type directly ("rila", "fia", "myga")
    or a product code ("6Y20B", "FIA5YR") which is mapped to its type
    using the Product Registry.

    Parameters
    ----------
    product_identifier : str
        Product code (e.g., "6Y20B") or product type ("rila")

    Returns
    -------
    ProductMethodology
        Methodology implementation

    Raises
    ------
    ValueError
        If product_identifier is not registered

    Examples
    --------
    >>> get_methodology("rila")  # Direct type
    <RILAMethodology>
    >>> get_methodology("6Y20B")  # Product code → mapped to "rila"
    <RILAMethodology>
    """
    # Import here to avoid circular imports
    from src.core.product_registry import is_product_code, get_product_type

    # If it's a product code, get the type from registry
    if is_product_code(product_identifier):
        product_type = get_product_type(product_identifier)
    else:
        product_type = product_identifier  # Assume it's already a type

    try:
        return BusinessRulesRegistry.get(product_type)
    except KeyError:
        available = BusinessRulesRegistry.list_registered()
        raise ValueError(
            f"Unknown product_type: '{product_type}' (from '{product_identifier}'). "
            f"Available: {', '.join(available)}"
        )


def get_adapter(adapter_type: str, **kwargs) -> DataSourceAdapter:
    """Get a data adapter.

    Convenience wrapper for AdapterRegistry.get().

    Parameters
    ----------
    adapter_type : str
        Adapter type: "aws", "local", "fixture"
    **kwargs
        Configuration arguments

    Returns
    -------
    DataSourceAdapter
        Configured adapter instance
    """
    return AdapterRegistry.get(adapter_type, **kwargs)


def get_aggregation_strategy(
    product_type: str, **kwargs
) -> AggregationStrategy:
    """Get aggregation strategy for a product type.

    Parameters
    ----------
    product_type : str
        Product type: "rila", "fia", "myga"
    **kwargs
        Strategy configuration

    Returns
    -------
    AggregationStrategy
        Configured strategy
    """
    return AggregationRegistry.get_for_product_type(product_type, **kwargs)


__all__ = [
    "BaseRegistry",
    "ConstraintRule",
    "BusinessRulesRegistry",
    "AdapterRegistry",
    "AggregationRegistry",
    "get_methodology",
    "get_adapter",
    "get_aggregation_strategy",
]
