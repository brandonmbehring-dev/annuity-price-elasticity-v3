"""
Backward Compatibility Module - Re-exports from src.config.builders.visualization_builders.

This file maintains backward compatibility after the Phase 2 reorganization.
All builders have been moved to src/config/builders/visualization_builders.py.

New code should import from:
    from src.config.builders.visualization_builders import ...
"""

# Re-export everything from the new location
from src.config.builders.visualization_builders import *  # noqa: F401,F403
