"""
Configuration modules for Annuity Price Elasticity v2.

Structure (Phase 2 Reorganization):
- types/: TypedDict configuration definitions
  - forecasting_config.py
  - pipeline_config.py
  - product_config.py
- builders/: Configuration builder functions
  - forecasting_builders.py
  - inference_builders.py
  - pipeline_builders.py
  - visualization_builders.py
- config_builder.py: Main orchestrator with re-exports
- configuration_validator.py: Config validation
- constants.py: Shared constants

Backward Compatibility:
- All previous import paths still work via re-exports
- from src.config.pipeline_config import ... → src.config.types.pipeline_config
- from src.config.pipeline_builders import ... → src.config.builders.pipeline_builders
"""

from src.config.constants import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_CONFIDENCE_LEVEL,
)
from src.config.types.product_config import (
    ProductConfig,
    get_product_config,
    get_wink_product_ids,
)

__all__ = [
    # Constants
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_CONFIDENCE_LEVEL",
    # Product config
    "ProductConfig",
    "get_product_config",
    "get_wink_product_ids",
]
