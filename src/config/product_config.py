"""
Backward Compatibility Module - Re-exports from src.config.types.product_config.

This file maintains backward compatibility after the Phase 2 reorganization.
All types have been moved to src/config/types/product_config.py.

New code should import from:
    from src.config.types.product_config import ...
"""

# Re-export everything from the new location
from src.config.types.product_config import *  # noqa: F401,F403
