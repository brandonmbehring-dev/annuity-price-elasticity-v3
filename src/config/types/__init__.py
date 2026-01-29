"""
Configuration Type Definitions.

Contains TypedDict classes for type-safe configuration management.
Extracted from flat src/config/ structure for better organization.

Modules:
- forecasting_config: Forecasting-specific TypedDicts
- pipeline_config: Pipeline-specific TypedDicts
- product_config: Product configuration and WINK mappings
"""

from src.config.types.forecasting_config import (
    ForecastingConfig,
    CrossValidationConfig,
    BootstrapModelConfig,
    BenchmarkModelConfig,
    PerformanceMonitoringConfig,
    ValidationFrameworkConfig,
    ForecastingInferenceConfig,
    RateScenarioConfig,
    ConfidenceIntervalConfig,
    TableauFormattingConfig,
    ProductMetadataConfig,
    ForecastingStageConfig,
    InferenceStageConfig,
)

from src.config.types.pipeline_config import (
    PipelineAWSConfig,
    S3Config,
    DataLoadingConfig,
    TDELoadingConfig,
    WINKLoadingConfig,
    PreprocessingTimeSeriesConfig,
    PreprocessingWINKConfig,
    PreprocessingTDEConfig,
    PreprocessingValidationConfig,
    ProductFilterConfig,
    SalesCleanupConfig,
    TimeSeriesConfig,
    WinkProcessingConfig,
    DataIntegrationConfig,
    WeeklyAggregationConfig,
    LagFeatureConfig,
    CompetitiveConfig,
    FeatureSelectionConfig,
    AICAnalysisConfig,
    FeatureValidationConfig,
    ModelSelectionConfig,
    FeatureSelectionStageConfig,
    VisualizationConfig,
    ExportConfig,
    BIExportConfig,
)

from src.config.types.product_config import (
    ProductConfig,
    PRODUCT_REGISTRY,
    get_product_config,
    get_default_product,
    get_wink_product_ids,
    get_pipeline_product_ids_as_lists,
    get_metadata_product_ids_as_lists,
    ProductFeatureConfig,
    get_default_feature_config,
)

__all__ = [
    # Forecasting types
    "ForecastingConfig",
    "CrossValidationConfig",
    "BootstrapModelConfig",
    "BenchmarkModelConfig",
    "PerformanceMonitoringConfig",
    "ValidationFrameworkConfig",
    "ForecastingInferenceConfig",
    "RateScenarioConfig",
    "ConfidenceIntervalConfig",
    "TableauFormattingConfig",
    "ProductMetadataConfig",
    "ForecastingStageConfig",
    "InferenceStageConfig",
    # Pipeline types
    "PipelineAWSConfig",
    "S3Config",
    "DataLoadingConfig",
    "ProductFilterConfig",
    "SalesCleanupConfig",
    "TimeSeriesConfig",
    "WinkProcessingConfig",
    "DataIntegrationConfig",
    "WeeklyAggregationConfig",
    "LagFeatureConfig",
    "CompetitiveConfig",
    "FeatureSelectionConfig",
    "VisualizationConfig",
    # Product config
    "ProductConfig",
    "PRODUCT_REGISTRY",
    "get_product_config",
    "get_wink_product_ids",
]
