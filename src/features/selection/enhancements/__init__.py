"""
Feature Selection Enhancements - Statistical Rigor Modules.

These modules provide advanced statistical techniques for enhanced feature selection:
- multiple_testing/: FWER/FDR corrections (Bonferroni, FDR)
- statistical_constraints/: CI-based constraint validation
- temporal_validation_engine: Time-series aware validation
- block_bootstrap_engine: Block bootstrap for time series
- out_of_sample_evaluation: Holdout evaluation framework

Integration: Controlled via FEATURE_FLAGS in interface_config.py.
All enhancements are disabled by default, can be enabled individually.
"""

# Block bootstrap for time series
from src.features.selection.enhancements.block_bootstrap_engine import (
    run_block_bootstrap_stability,
    create_temporal_blocks,
)

# Temporal validation
from src.features.selection.enhancements.temporal_validation_engine import (
    create_temporal_splits,
    evaluate_out_of_sample_performance,
    validate_temporal_structure,
)

# Out of sample evaluation
from src.features.selection.enhancements.out_of_sample_evaluation import (
    evaluate_temporal_generalization,
    run_time_series_cross_validation,
)

# Multiple testing subpackage
from src.features.selection.enhancements import multiple_testing

# Statistical constraints subpackage
from src.features.selection.enhancements import statistical_constraints

__all__ = [
    # Block bootstrap
    "run_block_bootstrap_stability",
    "create_temporal_blocks",
    # Temporal validation
    "create_temporal_splits",
    "evaluate_out_of_sample_performance",
    "validate_temporal_structure",
    # Out of sample evaluation
    "evaluate_temporal_generalization",
    "run_time_series_cross_validation",
    # Subpackages
    "multiple_testing",
    "statistical_constraints",
]
