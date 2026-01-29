"""
Enhanced configuration schemas using Pydantic for RILA pipeline.

This module provides type-safe configuration models that replace the existing
TypedDict patterns with runtime validation and better error messages.

Maintains backward compatibility with existing config_builder patterns.
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, Dict, Any
import numpy as np


class ForecastingConfigValidated(BaseModel):
    """
    Enhanced forecasting configuration with Pydantic validation.

    Replaces the TypedDict ForecastingConfig with runtime validation
    and maintains backward compatibility.
    """
    n_bootstrap_samples: int = Field(
        gt=0, le=1000,
        description="Number of bootstrap samples for ensemble forecasting"
    )
    ridge_alpha: float = Field(
        gt=0, le=100,
        description="Ridge regression regularization parameter"
    )
    random_state: int = Field(
        ge=0, le=2**32-1,
        description="Random seed for reproducibility"
    )
    exclude_holidays: bool = Field(
        default=True,
        description="Whether to exclude holiday periods from analysis"
    )
    mature_data_cutoff_days: int = Field(
        gt=0, le=365,
        description="Days to exclude recent data for maturity"
    )
    min_training_cutoff: int = Field(
        gt=5, le=100,
        description="Minimum observations required for training"
    )

    @validator('n_bootstrap_samples')
    def validate_bootstrap_samples(cls, v: int) -> int:
        if v < 10:
            raise ValueError('Bootstrap samples should be >= 10 for statistical validity')
        if v > 500:
            import warnings
            warnings.warn(f'Bootstrap samples {v} > 500 may be slow. Consider reducing.')
        return v

    @validator('ridge_alpha')
    def validate_ridge_alpha(cls, v: float) -> float:
        if v > 10:
            import warnings
            warnings.warn(f'Ridge alpha {v} is high and may cause underfitting')
        return v

    class Config:
        extra = "forbid"  # Prevent configuration typos
        validate_assignment = True  # Validate on attribute changes

    def dict(self) -> Dict[str, Any]:
        """Maintain compatibility with existing TypedDict usage."""
        return super().dict()


class BootstrapModelConfigValidated(BaseModel):
    """Enhanced bootstrap model configuration with validation."""
    estimator_type: Literal["Ridge", "Lasso", "ElasticNet"] = Field(
        default="Ridge",
        description="Type of linear regression estimator"
    )
    alpha: float = Field(
        gt=0, le=100,
        description="Regularization strength"
    )
    positive_constraint: bool = Field(
        default=True,
        description="Whether to enforce positive coefficients"
    )
    fit_intercept: bool = Field(
        default=True,
        description="Whether to fit intercept term"
    )
    normalize: bool = Field(
        default=False,
        description="Whether to normalize features (deprecated in sklearn)"
    )

    class Config:
        extra = "forbid"


class CrossValidationConfigValidated(BaseModel):
    """
    Enhanced cross-validation configuration with validation.

    NOTE: Time series cross-validation uses expanding window (TimeSeriesSplit) for all models.
    The validation_method field is for configuration compatibility but the implementation
    always uses expanding window, which is appropriate for time series data where we want
    to use all available historical data for training.
    """
    start_cutoff: int = Field(
        gt=5, le=200,
        description="First observation index for validation"
    )
    end_cutoff: Optional[int] = Field(
        default=None,
        description="Last observation index for validation (None = use all)"
    )
    validation_method: Literal["expanding_window", "rolling_window"] = Field(
        default="expanding_window",
        description="Time series validation method (implementation uses expanding_window)"
    )
    n_splits: int = Field(
        ge=0, le=50,
        description="Number of splits for time series validation"
    )

    @validator('end_cutoff')
    def validate_end_cutoff(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        if v is not None and 'start_cutoff' in values:
            if v <= values['start_cutoff']:
                raise ValueError('end_cutoff must be greater than start_cutoff')
        return v

    class Config:
        extra = "forbid"


# Backward compatibility functions for existing code
def build_forecasting_config_validated(**kwargs) -> ForecastingConfigValidated:
    """
    Build forecasting config with validation.

    Raises:
        ValueError: If configuration validation fails with clear error message
    """
    try:
        return ForecastingConfigValidated(**kwargs)
    except Exception as e:
        raise ValueError(f"Forecasting configuration validation failed: {e}")


def build_forecasting_config(**kwargs) -> Dict[str, Any]:
    """
    Legacy function for existing code - maintains TypedDict interface.

    This function provides backward compatibility while adding validation.
    """
    return build_forecasting_config_validated(**kwargs).dict()


# Example usage and validation demonstration
if __name__ == "__main__":
    print("Testing Pydantic configuration validation...")

    # Valid configuration
    try:
        config = ForecastingConfigValidated(
            n_bootstrap_samples=100,
            ridge_alpha=1.0,
            random_state=42,
            exclude_holidays=True,
            mature_data_cutoff_days=50,
            min_training_cutoff=30
        )
        print(f"✓ Valid configuration: {config.dict()}")
    except Exception as e:
        print(f"✗ Valid configuration failed: {e}")

    # Invalid configuration (should fail)
    try:
        invalid_config = ForecastingConfigValidated(
            n_bootstrap_samples=-10,  # Invalid: negative
            ridge_alpha=1.0,
            random_state=42
        )
        print(f"✗ Invalid configuration should have failed: {invalid_config}")
    except Exception as e:
        print(f"✓ Invalid configuration correctly failed: {e}")

    # Test backward compatibility
    try:
        legacy_dict = build_forecasting_config(
            n_bootstrap_samples=100,
            ridge_alpha=1.0,
            random_state=42,
            exclude_holidays=True,
            mature_data_cutoff_days=50,
            min_training_cutoff=30
        )
        print(f"✓ Legacy dict interface works: {type(legacy_dict)}")
    except Exception as e:
        print(f"✗ Legacy interface failed: {e}")

    print("Pydantic configuration validation test completed!")