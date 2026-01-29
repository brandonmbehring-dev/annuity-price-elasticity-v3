"""
Forecasting-specific TypedDict configuration definitions for clean_v3.

This module contains all forecasting-related configurations extracted from clean_v2,
including bootstrap forecasting, model parameters, and validation framework settings.

Following clean architecture principles:
- Domain-separated configuration management
- TypedDict for compile-time validation
- Clear separation from pipeline configurations
"""

from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from src.config.types.pipeline_config import VisualizationConfig


# =============================================================================
# Bootstrap Forecasting Configuration
# =============================================================================

class ForecastingConfig(TypedDict):
    """Configuration for bootstrap forecasting operations."""
    n_bootstrap_samples: int
    ridge_alpha: float
    random_state: int
    exclude_holidays: bool
    mature_data_cutoff_days: int
    min_training_cutoff: int


class CrossValidationConfig(TypedDict):
    """Configuration for time series cross-validation."""
    start_cutoff: int
    end_cutoff: Optional[int]
    validation_method: str  # 'holdout', 'expanding_window'
    n_splits: int


class BootstrapModelConfig(TypedDict):
    """Configuration for bootstrap Ridge regression model."""
    estimator_type: str  # 'Ridge', 'Lasso', 'ElasticNet'
    alpha: float
    positive_constraint: bool
    fit_intercept: bool
    n_estimators: int  # Number of bootstrap samples in ensemble
    normalize: bool


class BenchmarkModelConfig(TypedDict):
    """Configuration for benchmark forecasting model."""
    method: str  # 'rolling_average', 'naive', 'seasonal_naive'
    window_size: Optional[int]
    seasonal_period: Optional[int]


class PerformanceMonitoringConfig(TypedDict):
    """Configuration for performance monitoring and progress reporting."""
    progress_reporting_interval: int  # Report progress every N forecasts
    enable_detailed_logging: bool
    log_bootstrap_diagnostics: bool


class ValidationFrameworkConfig(TypedDict):
    """Configuration for mathematical validation framework."""
    tolerance_r2: float
    tolerance_mape: float
    tolerance_prediction: float
    enable_detailed_comparison: bool
    reference_results_path: str


# =============================================================================
# Inference Configuration (for RILA Price Elasticity Analysis with Forecasting)
# =============================================================================

class ForecastingInferenceConfig(TypedDict):
    """Configuration for RILA price elasticity inference operations with forecasting.

    Comprehensive configuration capturing all hardcoded values from original notebook
    to enable full configurability and mathematical equivalence validation.

    Note: For general inference config, use `src.core.types.InferenceConfig` instead.
    This detailed config is for forecasting workflows with specific ML parameters.
    """

    # Model hyperparameters
    n_estimators: int
    weight_decay_factor: float
    random_state: int
    ridge_alpha: float

    # Business parameters
    sales_multiplier: float
    momentum_lookback_periods: int
    training_cutoff_days: int  # Days before current date for training cutoff

    # Feature configuration
    target_variable: str
    features: List[str]

    # Data filtering parameters
    exclude_zero_sales: bool
    date_filter_start: str  # Format: "YYYY-MM-DD"


# Deprecated alias for backward compatibility - use ForecastingInferenceConfig
InferenceConfig = ForecastingInferenceConfig


class RateScenarioConfig(TypedDict):
    """Configuration for rate scenario analysis."""
    rate_min: float
    rate_max: float
    rate_steps: int
    competitor_rate_adjustment: float


class ConfidenceIntervalConfig(TypedDict):
    """Configuration for confidence interval calculations."""
    confidence_level: float
    rounding_precision: int
    basis_points_multiplier: int


class TableauFormattingConfig(TypedDict):
    """Configuration for Tableau output formatting."""
    prudential_rate_col: str
    competitor_rate_col: str
    sales_lag_cols: List[str]
    sales_rounding_power: int


class ProductMetadataConfig(TypedDict):
    """Configuration for product metadata and business information."""
    product_name: str  # e.g., "FlexGuard_6Y20B"
    version: str  # e.g., "v2_1"
    flexguard_product_ids: Dict[str, List[int]]  # Company to product ID mapping
    product_name_dict: Dict[str, str]  # Company to full product name mapping


# VisualizationConfig imported via TYPE_CHECKING at module top to avoid circular import


# =============================================================================
# Combined Forecasting Stage Configuration
# =============================================================================

class ForecastingStageConfig(TypedDict):
    """Combined configuration for complete forecasting stage."""
    forecasting_config: ForecastingConfig
    cv_config: CrossValidationConfig
    bootstrap_model_config: BootstrapModelConfig
    benchmark_model_config: BenchmarkModelConfig
    performance_monitoring_config: PerformanceMonitoringConfig
    validation_config: ValidationFrameworkConfig
    model_features: List[str]  # Competitive intelligence features for Ridge model
    benchmark_features: List[str]  # Features for benchmark model (persistence)
    target_variable: str  # Target variable for prediction
    model_sign_correction_config: Dict[str, Any]  # Sign correction for model features
    benchmark_sign_correction_config: Dict[str, Any]  # Sign correction for benchmark features


class InferenceStageConfig(TypedDict):
    """Combined configuration for inference stage.

    Comprehensive configuration for RILA price elasticity inference pipeline
    that eliminates all hardcoded values from original notebook.
    """
    inference_config: ForecastingInferenceConfig
    rate_scenario_config: RateScenarioConfig
    confidence_interval_config: ConfidenceIntervalConfig
    tableau_formatting_config: TableauFormattingConfig
    product_metadata_config: ProductMetadataConfig
    visualization_config: 'VisualizationConfig'  # String annotation to avoid circular import