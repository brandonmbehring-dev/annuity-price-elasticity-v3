"""
Pipeline-specific TypedDict configuration definitions for clean_v3.

This module contains all pipeline-related configurations extracted from clean_v1,
including AWS operations, data loading, preprocessing, and feature engineering.
Forecasting configurations are separated into forecasting_config.py.

Following clean architecture principles:
- Domain-separated configuration management
- TypedDict for compile-time validation
- Clear separation from forecasting configurations
"""

from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Import forecasting configs to avoid duplication
from src.config.types.forecasting_config import (
    ForecastingConfig, CrossValidationConfig, BootstrapModelConfig,
    BenchmarkModelConfig, ValidationFrameworkConfig, ForecastingStageConfig
)


# =============================================================================
# AWS & Infrastructure Configuration
# =============================================================================

class PipelineAWSConfig(TypedDict):
    """Configuration for full pipeline AWS operations with separate source/output buckets.

    Note: For simple single-bucket operations, use `src.core.types.AWSConfig` instead.
    This extended config is for pipeline workflows requiring separate source and output paths.
    """

    xid: str
    role_arn: str
    sts_endpoint_url: str
    source_bucket_name: str
    output_bucket_name: str
    output_base_path: str


# Deprecated alias for backward compatibility - use PipelineAWSConfig
AWSConfig = PipelineAWSConfig


class S3Config(TypedDict):
    """Configuration for S3 data operations."""
    bucket_name: str
    base_path: str
    current_date: str
    date_path: str
    export_enabled: bool


# =============================================================================
# Data Loading & Extraction Configuration
# =============================================================================

class DataLoadingConfig(TypedDict):
    """Configuration for data loading operations."""
    purpose: str  # 'validation', 'features', 'forecasting'
    version: int
    date_filter_start: str
    days_before_mature: int
    exclude_holidays: bool


class TDELoadingConfig(TypedDict):
    """Configuration for TDE data loading and processing."""
    file_path: str
    required_columns: List[str]
    column_mapping: Dict[str, str]
    date_column: str
    start_date: str
    end_date: str


class WINKLoadingConfig(TypedDict):
    """Configuration for WINK competitive data loading."""
    data_source: str
    product_filter: List[str]
    rate_type: str  # 'mid', 'spread', 'all'
    product_ids: Dict[str, List[int]]
    rolling_days: int


# =============================================================================
# Preprocessing Configuration
# =============================================================================

class PreprocessingTimeSeriesConfig(TypedDict):
    """Time series parameters for data cleaning stage."""
    rolling_window_days: int
    groupby_frequency: str
    analysis_start_date: str
    interpolation_method: str  # 'linear', 'forward', 'backward'


class PreprocessingWINKConfig(TypedDict):
    """WINK data standardization parameters."""
    company_column: str
    standardization_mapping: Dict[str, str]
    product_column: str
    hierarchy_mapping: Dict[str, str]
    rate_columns: List[str]
    decimal_places: int


class PreprocessingTDEConfig(TypedDict):
    """TDE data cleaning and validation parameters."""
    premium_column: str
    min_premium: float
    max_premium: float
    business_rules: Dict[str, Any]
    numeric_columns: List[str]
    categorical_columns: List[str]


class PreprocessingValidationConfig(TypedDict):
    """Data validation parameters for preprocessing stage."""
    min_completeness: float
    check_duplicates: bool
    required_columns: List[str]
    expected_dtypes: Optional[Dict[str, str]]


# =============================================================================
# Processing & Business Logic Configuration
# =============================================================================

class ProcessingMathematicalConfig(TypedDict):
    """Mathematical constants and calculations for business logic."""
    weight_decay_factor: float
    max_rate_change: float
    confidence_level: float
    threshold_values: Dict[str, float]


class ProcessingTimeSeriesConfig(TypedDict):
    """Time series parameters for business logic stage."""
    seasonality_period: int
    trend_window: int
    lag_periods: List[int]
    window_sizes: List[int]
    statistics: List[str]  # 'mean', 'std', 'min', 'max'


class ProcessingCompetitiveConfig(TypedDict):
    """Competitive analysis parameters."""
    benchmark_company: str
    target_company: str
    rate_column: str
    company_column: str
    spread_calculation_method: str


class ProcessingFilteringConfig(TypedDict):
    """Data filtering parameters for business logic."""
    buffer_rate: str
    term: str
    product_name: str
    date_range: Dict[str, str]
    quantile_threshold: float


# =============================================================================
# Feature Selection Configuration
# =============================================================================

class FeatureSelectionConfig(TypedDict):
    """AIC-based feature selection parameters for RILA price elasticity analysis."""
    target_variable: str
    base_features: List[str]
    candidate_features: List[str]
    max_candidate_features: int
    economic_constraints: bool
    min_observations: int
    validation_split: float


class AICAnalysisConfig(TypedDict):
    """Configuration for AIC model selection and validation."""
    max_models_to_test: int
    convergence_tolerance: float
    r_squared_threshold: float
    p_value_threshold: float
    coefficient_constraints: Dict[str, str]  # {'competitor_': 'negative', 'prudential_rate': 'positive'}


class FeatureValidationConfig(TypedDict):
    """Feature validation and constraint checking parameters."""
    require_competitor_features: bool
    require_prudential_features: bool
    max_correlation_threshold: float
    min_feature_importance: float
    cross_validation_folds: int


class ModelSelectionConfig(TypedDict):
    """Model selection criteria and ranking parameters."""
    selection_criteria: str  # 'aic', 'bic', 'cross_validation'
    ensemble_size: int
    bootstrap_iterations: int
    confidence_level: float
    performance_metrics: List[str]  # ['aic', 'r_squared', 'mape', 'rmse']


# =============================================================================
# Feature Engineering Configuration
# =============================================================================

class EngineeringTimeSeriesConfig(TypedDict):
    """Feature engineering time series parameters."""
    max_lag_periods: int
    weekly_aggregation_freq: str
    mature_data_offset_days: int
    lag_direction: str  # 'backward', 'forward', 'both'
    # Clean_v3 competitive feature configuration
    company_columns: List[str]
    min_companies_required: int
    enable_fallback_strategy: bool


class EngineeringMathematicalConfig(TypedDict):
    """Feature engineering mathematical parameters."""
    polynomial_degree: int
    handle_zeros: str  # 'add_constant', 'remove', 'log1p'
    standardization_method: str  # 'zscore', 'minmax'
    bin_edges: List[float]
    bin_labels: Optional[List[str]]


class EngineeringInteractionConfig(TypedDict):
    """Feature interaction and combination parameters."""
    feature_pairs: List[tuple]
    base_columns: List[str]
    numerator_columns: List[str]
    denominator_columns: List[str]
    group_columns: List[str]
    value_columns: List[str]
    agg_functions: List[str]


class EngineeringTemporalConfig(TypedDict):
    """Temporal feature engineering parameters."""
    date_column: str
    reference_date: Optional[str]
    holiday_start_day: int
    holiday_end_day: int
    business_day_reference: str


# =============================================================================
# Model Training & Validation Configuration
# =============================================================================

class ModelTrainingConfig(TypedDict):
    """Model training parameters."""
    alpha: float
    positive_constraint: bool
    n_estimators: int
    random_state: int
    estimator_params: Dict[str, Any]


class ModelValidationConfig(TypedDict):
    """Model validation parameters."""
    train_end_date: str
    validation_days: int
    n_splits: int
    confidence_level: float
    sample_weight_column: Optional[str]


class ModelPerformanceConfig(TypedDict):
    """Model performance evaluation parameters."""
    business_thresholds: Dict[str, float]
    r2_min: float
    mape_max: float
    ae_ratio_min: float
    ae_ratio_max: float


# =============================================================================
# Forecasting Configuration
# =============================================================================

class PipelineForecastingConfig(TypedDict):
    """Pipeline forecasting model parameters (for pipeline stage configuration)."""
    cutoff_start: int
    cutoff_end: int
    target_column: str
    features: List[str]
    ridge_alpha: float
    n_estimators: int


class ForecastingValidationConfig(TypedDict):
    """Forecasting validation parameters."""
    performance_thresholds: Dict[str, float]
    comparison_tolerance: float
    r2_excellent: float
    r2_good: float
    mape_acceptable: float


# =============================================================================
# Visualization Configuration
# =============================================================================

class VisualizationConfig(TypedDict):
    """Chart formatting and display parameters."""
    figure_width: int
    figure_height: int
    plot_style: str
    plot_palette: str
    line_width: int
    marker_size: int
    grid_alpha: float
    confidence_alpha: float


class PlotStyleConfig(TypedDict):
    """Plot styling parameters."""
    figure_size: tuple
    suptitle: str
    suptitle_fontsize: int
    linewidth: int
    qr_color: str
    ensemble_color: str
    actual_color: str


class AnnotationConfig(TypedDict):
    """Annotation parameters for plots."""
    position: tuple
    fontsize: int
    verticalalignment: str
    bbox_style: str
    bbox_facecolor: str
    bbox_alpha: float


# =============================================================================
# Export & Output Configuration
# =============================================================================

class ExportConfig(TypedDict):
    """Export configuration parameters."""
    export_path: str
    export_format: str  # 'csv', 'parquet', 'excel'
    include_metadata: bool
    round_decimals: bool
    decimal_places: int
    include_summary: bool


class BIExportConfig(TypedDict):
    """BI team export configuration."""
    filename_base: str
    current_date: str
    output_directory: str
    version: str
    product: str
    model_start_date: str
    forecasting_date: str
    confidence_level: int


# =============================================================================
# Memory Management Configuration
# =============================================================================

class MemoryConfig(TypedDict):
    """Memory management parameters."""
    cleanup_enabled: bool
    gc_frequency: int
    log_memory_usage: bool
    step_name: str


# =============================================================================
# Progress Logging Configuration
# =============================================================================

class LoggingConfig(TypedDict):
    """Progress logging parameters."""
    log_level: str
    step_descriptions: List[str]
    result_info: str
    summary_info: str


# =============================================================================
# Combined Configuration Types for Pipeline Stages
# =============================================================================

class ExtractionStageConfig(TypedDict):
    """Combined configuration for extraction stage."""
    aws_config: AWSConfig
    data_loading_config: DataLoadingConfig
    tde_loading_config: Optional[TDELoadingConfig]
    wink_loading_config: Optional[WINKLoadingConfig]
    logging_config: LoggingConfig


class PreprocessingStageConfig(TypedDict):
    """Combined configuration for preprocessing stage."""
    time_series_config: PreprocessingTimeSeriesConfig
    wink_config: PreprocessingWINKConfig
    tde_config: PreprocessingTDEConfig
    validation_config: PreprocessingValidationConfig
    memory_config: MemoryConfig


class ProcessingStageConfig(TypedDict):
    """Combined configuration for processing stage."""
    mathematical_config: ProcessingMathematicalConfig
    time_series_config: ProcessingTimeSeriesConfig
    competitive_config: ProcessingCompetitiveConfig
    filtering_config: ProcessingFilteringConfig


class FeatureSelectionStageConfig(TypedDict):
    """Combined configuration for feature selection stage."""
    selection_config: FeatureSelectionConfig
    analysis_config: AICAnalysisConfig
    validation_config: FeatureValidationConfig
    model_selection_config: ModelSelectionConfig


class EngineeringStageConfig(TypedDict):
    """Combined configuration for engineering stage."""
    time_series_config: EngineeringTimeSeriesConfig
    mathematical_config: EngineeringMathematicalConfig
    interaction_config: EngineeringInteractionConfig
    temporal_config: EngineeringTemporalConfig


class VisualizationStageConfig(TypedDict):
    """Combined configuration for visualization stage."""
    visualization_config: VisualizationConfig
    plot_style_config: PlotStyleConfig
    annotation_config: AnnotationConfig
    export_config: ExportConfig


# =============================================================================
# Clean_v6 Pipeline Convenience Function Configuration
# =============================================================================

class ProductFilterConfig(TypedDict):
    """Configuration for product filtering pipeline."""
    product_name: str
    buffer_rate: str
    term: str

class SalesCleanupConfig(TypedDict):
    """Configuration for sales data cleanup pipeline."""
    min_premium: float
    max_premium: float
    quantile_threshold: float
    start_date_col: str
    end_date_col: str
    processing_days_col: str
    premium_column: str
    sales_alias_col: str

class TimeSeriesConfig(TypedDict):
    """Configuration for time series creation pipeline."""
    date_column: str
    value_column: str
    alias_date_col: str
    alias_value_col: str
    groupby_frequency: str
    rolling_window_days: int

class WinkProcessingConfig(TypedDict):
    """Configuration for WINK competitive rate processing pipeline."""
    product_ids: Dict[str, List[int]]
    start_date: str
    rolling_days: int
    buffer_modifier_filter: str
    indexing_method_filter: str
    crediting_frequency_filter: str
    actuarial_view_filter: str
    product_type_filter: str
    participation_rate_target: float
    index_name_filter: str
    buffer_rates_allowed: List[float]
    default_cap_rate: float
    max_cap_rate: float
    data_filter_start_date: str

class DataIntegrationConfig(TypedDict):
    """Configuration for daily data integration pipeline (split from IntegrationConfig)."""
    # Data integration parameters
    start_date: str
    end_date: str
    cpi_column: str
    sales_columns: List[str]
    economic_columns: List[str]
    rolling_window: int
    # Business counter parameters
    sales_reference_column: str
    business_counter_column: str

class WeeklyAggregationConfig(TypedDict):
    """Configuration for weekly aggregation pipeline (split from IntegrationConfig)."""
    # Weekly aggregation parameters
    weekly_aggregation_freq: str
    weekly_agg_dict: Dict[str, str]

class LagFeatureConfig(TypedDict):
    """Configuration for lag feature engineering pipeline."""
    lag_column_configs: List[Dict[str, Any]]
    polynomial_base_columns: List[str]
    max_lag_periods: int
    allow_inplace_operations: bool

class CompetitiveConfig(TypedDict):
    """Configuration for competitive feature engineering pipeline."""
    company_columns: List[str]
    min_companies_required: int

class FeatureConfig(TypedDict):
    """Configuration for final feature preparation pipeline."""
    feature_analysis_start_date: str
    date_column: str
    day_of_year_column: str
    holiday_column: str
    holiday_start_day: int
    holiday_end_day: int
    prudential_rate_column: str
    competitor_rate_column: str
    spread_column: str
    log_transform_source_column: str
    log_transform_target_column: str
    mature_data_cutoff_date: str


# =============================================================================
# Forecasting & Inference Configuration (imports from forecasting_config.py)
# =============================================================================
# Note: Individual configs (ForecastingConfig, CrossValidationConfig, etc.)
# are imported from src.config.forecasting_config to avoid duplication


# =============================================================================
# Master Pipeline Configuration
# =============================================================================

class PipelineConfig(TypedDict):
    """Master configuration for entire pipeline."""
    extraction: ExtractionStageConfig
    preprocessing: PreprocessingStageConfig
    processing: ProcessingStageConfig
    feature_selection: FeatureSelectionStageConfig
    engineering: EngineeringStageConfig
    visualization: VisualizationStageConfig
    model_training: ModelTrainingConfig
    model_validation: ModelValidationConfig
    forecasting: ForecastingStageConfig
    bi_export: BIExportConfig