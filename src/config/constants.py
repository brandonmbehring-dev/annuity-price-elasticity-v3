"""
Constants for Annuity Price Elasticity v2.

Central repository of default values and magic numbers extracted for maintainability.
Use these constants instead of hardcoding values in multiple places.

Usage:
    from src.config.constants import (
        DEFAULT_RANDOM_SEED,
        DEFAULT_CONFIDENCE_LEVEL,
        DEFAULT_N_BOOTSTRAP,
    )
"""

# =============================================================================
# STATISTICAL DEFAULTS
# =============================================================================

# Random seed for reproducibility across all stochastic operations
DEFAULT_RANDOM_SEED: int = 42

# Default number of bootstrap samples
DEFAULT_N_BOOTSTRAP: int = 100

# Default confidence level for statistical inference
DEFAULT_CONFIDENCE_LEVEL: float = 0.95

# Confidence levels commonly used in reporting
REPORTING_CONFIDENCE_LEVELS: tuple = (0.80, 0.90, 0.95)


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Ridge regression regularization (alpha)
DEFAULT_RIDGE_ALPHA: float = 1.0

# Number of estimators for bagging/ensemble models
DEFAULT_N_ESTIMATORS: int = 100


# =============================================================================
# DATA QUALITY THRESHOLDS
# =============================================================================

# Minimum observations required for reliable inference
MIN_OBSERVATIONS: int = 30

# Maximum acceptable VIF for multicollinearity
MAX_VIF_THRESHOLD: float = 10.0

# R-squared warning threshold (suspect leakage above this)
R_SQUARED_WARNING_THRESHOLD: float = 0.95

# R-squared halt threshold (halt pipeline above this)
R_SQUARED_HALT_THRESHOLD: float = 0.99


# =============================================================================
# FEATURE SELECTION
# =============================================================================

# Maximum number of candidate models to evaluate
DEFAULT_MAX_CANDIDATE_MODELS: int = 100

# Number of models to include in bootstrap analysis
DEFAULT_MODELS_TO_ANALYZE: int = 10


# =============================================================================
# NUMERICAL PRECISION
# =============================================================================

# Tolerance for mathematical equivalence validation
NUMERICAL_TOLERANCE: float = 1e-12

# Rounding precision for output tables
DEFAULT_DECIMAL_PLACES: int = 4


__all__ = [
    # Statistical defaults
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_CONFIDENCE_LEVEL",
    "REPORTING_CONFIDENCE_LEVELS",
    # Model hyperparameters
    "DEFAULT_RIDGE_ALPHA",
    "DEFAULT_N_ESTIMATORS",
    # Data quality thresholds
    "MIN_OBSERVATIONS",
    "MAX_VIF_THRESHOLD",
    "R_SQUARED_WARNING_THRESHOLD",
    "R_SQUARED_HALT_THRESHOLD",
    # Feature selection
    "DEFAULT_MAX_CANDIDATE_MODELS",
    "DEFAULT_MODELS_TO_ANALYZE",
    # Numerical precision
    "NUMERICAL_TOLERANCE",
    "DEFAULT_DECIMAL_PLACES",
]
