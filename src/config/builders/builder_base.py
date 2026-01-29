"""
Generic Builder Patterns for Configuration Construction.

This module provides reusable patterns for building configuration objects,
reducing repetition across the various builder modules.

Usage:
    from src.config.builders.builder_base import (
        simple_config_builder,
        aggregate_configs,
        with_defaults,
    )

Design Principles:
- DRY: Common patterns extracted to reduce duplication
- Type Safety: Generic typing for config construction
- Composability: Builders can be combined and chained
"""

from typing import TypeVar, Dict, Any, Callable, Optional, List, Union

T = TypeVar('T')


def simple_config_builder(config_class: type[T], **kwargs) -> T:
    """Generic wrapper for TypedDict/dataclass construction.

    Provides a unified pattern for constructing configuration objects
    from keyword arguments.

    Parameters
    ----------
    config_class : type[T]
        The configuration class to instantiate
    **kwargs : Any
        Configuration parameters

    Returns
    -------
    T
        Instantiated configuration object

    Examples
    --------
    >>> config = simple_config_builder(ForecastingConfig, n_samples=1000)
    """
    return config_class(kwargs)


def aggregate_configs(
    builders: Dict[str, Callable[[], Dict[str, Any]]]
) -> Dict[str, Any]:
    """Aggregate multiple config builders into single dict.

    Useful for combining related configuration sections into
    a unified configuration object.

    Parameters
    ----------
    builders : Dict[str, Callable[[], Dict[str, Any]]]
        Mapping of section names to builder functions

    Returns
    -------
    Dict[str, Any]
        Combined configuration dictionary

    Examples
    --------
    >>> configs = aggregate_configs({
    ...     'model': build_model_config,
    ...     'validation': build_validation_config,
    ... })
    """
    return {name: builder() for name, builder in builders.items()}


def with_defaults(
    user_config: Dict[str, Any],
    defaults: Dict[str, Any],
    deep_merge: bool = False
) -> Dict[str, Any]:
    """Merge user configuration with defaults.

    User-provided values take precedence over defaults.

    Parameters
    ----------
    user_config : Dict[str, Any]
        User-provided configuration overrides
    defaults : Dict[str, Any]
        Default configuration values
    deep_merge : bool, default=False
        If True, recursively merge nested dictionaries

    Returns
    -------
    Dict[str, Any]
        Merged configuration

    Examples
    --------
    >>> config = with_defaults({'n_samples': 500}, DEFAULT_BOOTSTRAP_CONFIG)
    """
    if not deep_merge:
        return {**defaults, **user_config}

    result = defaults.copy()
    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = with_defaults(value, result[key], deep_merge=True)
        else:
            result[key] = value
    return result


def build_config_section(
    section_name: str,
    config_class: type[T],
    **kwargs
) -> Dict[str, T]:
    """Build a named configuration section.

    Convenience function for building configurations that need to be
    wrapped with a section name.

    Parameters
    ----------
    section_name : str
        Name for the configuration section
    config_class : type[T]
        Configuration class to instantiate
    **kwargs : Any
        Configuration parameters

    Returns
    -------
    Dict[str, T]
        Dictionary with section_name as key

    Examples
    --------
    >>> section = build_config_section('bootstrap', BootstrapConfig, n_samples=1000)
    {'bootstrap': BootstrapConfig({'n_samples': 1000})}
    """
    return {section_name: config_class(kwargs)}


def validate_config_keys(
    config: Dict[str, Any],
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None
) -> List[str]:
    """Validate configuration has required keys.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration to validate
    required_keys : List[str]
        Keys that must be present
    optional_keys : Optional[List[str]], default=None
        Keys that are allowed but not required

    Returns
    -------
    List[str]
        List of validation error messages (empty if valid)

    Examples
    --------
    >>> errors = validate_config_keys(config, ['n_samples', 'random_state'])
    """
    errors = []
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")

    if optional_keys is not None:
        allowed_keys = set(required_keys) | set(optional_keys)
        for key in config:
            if key not in allowed_keys:
                errors.append(f"Unknown configuration key: {key}")

    return errors


def chain_builders(
    *builders: Callable[[Dict[str, Any]], Dict[str, Any]],
    initial: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Chain multiple config builders together.

    Each builder receives the output of the previous builder,
    allowing for progressive configuration construction.

    Parameters
    ----------
    *builders : Callable[[Dict[str, Any]], Dict[str, Any]]
        Builder functions to chain
    initial : Optional[Dict[str, Any]], default=None
        Initial configuration to start with

    Returns
    -------
    Dict[str, Any]
        Final configuration after all builders applied

    Examples
    --------
    >>> config = chain_builders(
    ...     add_model_config,
    ...     add_validation_config,
    ...     initial=base_config
    ... )
    """
    result = initial if initial is not None else {}
    for builder in builders:
        result = builder(result)
    return result


class ConfigBuilderMixin:
    """Mixin class providing common builder functionality.

    Inherit from this class to add standard builder methods
    to configuration builder classes.
    """

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.

        Later configs override earlier ones.
        """
        result: Dict[str, Any] = {}
        for config in configs:
            result.update(config)
        return result

    @staticmethod
    def filter_none_values(config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from configuration."""
        return {k: v for k, v in config.items() if v is not None}

    @staticmethod
    def add_metadata(
        config: Dict[str, Any],
        version: str = "v2",
        **metadata: Any
    ) -> Dict[str, Any]:
        """Add metadata section to configuration."""
        config['_metadata'] = {
            'version': version,
            **metadata
        }
        return config


__all__ = [
    "simple_config_builder",
    "aggregate_configs",
    "with_defaults",
    "build_config_section",
    "validate_config_keys",
    "chain_builders",
    "ConfigBuilderMixin",
]
