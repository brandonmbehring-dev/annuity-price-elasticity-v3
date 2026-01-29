"""
Modeling modules for RILA price elasticity analysis.

Modules:
- inference: Price elasticity inference functions (canonical)
- inference_validation: Input validation functions (extracted)
- forecasting: Bootstrap forecasting and time series operations

Canonical Imports:
    from src.models.inference import center_baseline, rate_adjustments
    from src.models.inference_validation import validate_center_baseline_inputs
"""

# Inference functions
from src.models.inference import (
    TrainingData,
    prepare_training_data,
    train_bootstrap_model,
    center_baseline,
    rate_adjustments,
    confidence_interval,
    melt_dataframe_for_tableau
)

# Validation functions (extracted module)
from src.models.inference_validation import (
    validate_center_baseline_inputs,
    validate_rate_adjustments_inputs,
    validate_confidence_interval_inputs,
    validate_melt_dataframe_inputs
)

__all__ = [
    # Inference
    "TrainingData",
    "prepare_training_data",
    "train_bootstrap_model",
    "center_baseline",
    "rate_adjustments",
    "confidence_interval",
    "melt_dataframe_for_tableau",
    # Validation
    "validate_center_baseline_inputs",
    "validate_rate_adjustments_inputs",
    "validate_confidence_interval_inputs",
    "validate_melt_dataframe_inputs",
]
