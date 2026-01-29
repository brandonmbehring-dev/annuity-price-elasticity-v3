"""
Feature Selection Engines - Core Computational Modules.

These are the atomic computational engines for feature selection:
- AIC Engine: Information criterion-based model comparison
- Bootstrap Engine: Stability analysis via resampling
- Constraints Engine: Economic constraint validation
- RidgeCV Engine: Cross-validated Ridge feature selection (TD-05 alternative)

All engines follow functional design with immutable inputs/outputs.
"""

# AIC-based feature evaluation
from src.features.selection.engines.aic_engine import (
    evaluate_aic_combinations,
    calculate_aic_for_features,
    generate_feature_combinations,
)

# Bootstrap stability analysis
from src.features.selection.engines.bootstrap_engine import (
    run_bootstrap_stability,
    calculate_bootstrap_metrics,
    assess_model_stability,
)

# Economic constraint validation
from src.features.selection.engines.constraints_engine import (
    apply_economic_constraints,
    validate_constraint_rule,
    generate_constraint_violations,
)

# RidgeCV feature selection (TD-05 alternative)
from src.features.selection.engines.ridge_cv_engine import (
    RidgeCVConfig,
    RidgeCVResults,
    evaluate_ridge_cv_combinations,
    compare_with_aic_selection,
)

__all__ = [
    # AIC Engine
    "evaluate_aic_combinations",
    "calculate_aic_for_features",
    "generate_feature_combinations",
    # Bootstrap Engine
    "run_bootstrap_stability",
    "calculate_bootstrap_metrics",
    "assess_model_stability",
    # Constraints Engine
    "apply_economic_constraints",
    "validate_constraint_rule",
    "generate_constraint_violations",
    # RidgeCV Engine
    "RidgeCVConfig",
    "RidgeCVResults",
    "evaluate_ridge_cv_combinations",
    "compare_with_aic_selection",
]
