"""
Configuration Builder Functions.

Contains builder functions that construct configuration dictionaries.
Extracted from flat src/config/ structure for better organization.

Modules:
- builder_base: Generic builder patterns and utilities
- defaults: Centralized default values
- forecasting_builders: Forecasting stage configurations
- inference_builders: Inference/metadata configurations
- pipeline_builders: Pipeline stage configurations
- visualization_builders: Visualization configurations
"""

# Generic builder patterns (TD-08)
from src.config.builders.builder_base import (
    simple_config_builder,
    aggregate_configs,
    with_defaults,
    build_config_section,
    validate_config_keys,
    chain_builders,
    ConfigBuilderMixin,
)

# Centralized defaults (TD-08)
from src.config.builders.defaults import (
    DEFAULTS,
    get_default,
    get_product_defaults,
    get_feature_list,
    # Specific default collections
    BOOTSTRAP_DEFAULTS,
    INFERENCE_DEFAULTS,
    COMPETITIVE_FEATURES,
)

from src.config.builders.forecasting_builders import (
    build_forecasting_stage_config,
)

from src.config.builders.inference_builders import (
    build_inference_config,
    build_inference_stage_config,
    build_product_metadata_config,
    build_rate_scenario_config,
    build_confidence_interval_config,
    build_tableau_formatting_config,
)

from src.config.builders.pipeline_builders import (
    get_lag_column_configs,
    get_weekly_aggregation_dict,
    build_pipeline_configs,
    build_pipeline_configs_for_product,
)

from src.config.builders.visualization_builders import (
    build_visualization_config,
)

__all__ = [
    # Generic builder patterns (TD-08)
    "simple_config_builder",
    "aggregate_configs",
    "with_defaults",
    "build_config_section",
    "validate_config_keys",
    "chain_builders",
    "ConfigBuilderMixin",
    # Centralized defaults (TD-08)
    "DEFAULTS",
    "get_default",
    "get_product_defaults",
    "get_feature_list",
    "BOOTSTRAP_DEFAULTS",
    "INFERENCE_DEFAULTS",
    "COMPETITIVE_FEATURES",
    # Forecasting builders
    "build_forecasting_stage_config",
    # Inference builders
    "build_inference_config",
    "build_inference_stage_config",
    "build_product_metadata_config",
    "build_rate_scenario_config",
    "build_confidence_interval_config",
    "build_tableau_formatting_config",
    # Pipeline builders
    "get_lag_column_configs",
    "get_weekly_aggregation_dict",
    "build_pipeline_configs",
    "build_pipeline_configs_for_product",
    # Visualization builders
    "build_visualization_config",
]
