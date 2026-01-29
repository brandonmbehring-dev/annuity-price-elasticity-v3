"""
Shared Type Definitions for Annuity Price Elasticity v2.

Central repository of TypedDicts and type aliases used across modules.
Following CODING_STANDARDS.md: All configurations use TypedDict for type safety.

Usage:
    from src.core.types import InferenceConfig, AWSConfig, FeatureConfig
"""

from typing import TypedDict, List, Tuple, Optional, Dict, Any


# =============================================================================
# AWS CONFIGURATION
# =============================================================================


class AWSConfig(TypedDict):
    """AWS configuration for simple single-bucket S3 operations.

    This is the canonical AWS config for adapters using a single bucket.
    For pipeline workflows requiring separate source/output buckets, use
    `src.config.pipeline_config.PipelineAWSConfig` instead.

    Attributes
    ----------
    sts_endpoint_url : str
        STS service endpoint URL
    role_arn : str
        IAM role ARN to assume for S3 access
    xid : str
        User/session identifier for role assumption
    bucket_name : str
        S3 bucket name for data operations
    """

    sts_endpoint_url: str
    role_arn: str
    xid: str
    bucket_name: str


class AWSConfigOptional(TypedDict, total=False):
    """Optional AWS configuration fields."""

    region: str
    session_duration: int


# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================


class InferenceConfig(TypedDict):
    """Configuration for price elasticity inference.

    Attributes
    ----------
    product_code : str
        Product identifier (e.g., "6Y20B")
    product_type : str
        Product category: "rila", "fia", or "myga"
    own_rate_column : str
        Column name for own (Prudential) rate
    competitor_rate_column : str
        Column name for competitor rate aggregate
    target_column : str
        Target variable for modeling (e.g., "sales_target_current")
    rate_adjustment_range : Tuple[int, int]
        Min/max basis points for rate adjustments (e.g., (-300, 300))
    n_bootstrap : int
        Number of bootstrap iterations
    confidence_levels : List[float]
        Confidence intervals to compute (e.g., [0.80, 0.90, 0.95])
    """

    product_code: str
    product_type: str
    own_rate_column: str
    competitor_rate_column: str
    target_column: str
    rate_adjustment_range: Tuple[int, int]
    n_bootstrap: int
    confidence_levels: List[float]


class InferenceConfigOptional(TypedDict, total=False):
    """Optional inference configuration fields."""

    random_seed: int
    parallel_jobs: int
    verbose: bool


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================


class FeatureConfig(TypedDict):
    """Configuration for feature engineering and selection.

    Attributes
    ----------
    base_features : List[str]
        Features always included in models
    candidate_features : List[str]
        Features to evaluate for selection
    target_variable : str
        Target variable name
    max_lag : int
        Maximum lag periods for time series features
    analysis_start_date : str
        Start date for analysis (ISO format)
    """

    base_features: List[str]
    candidate_features: List[str]
    target_variable: str
    max_lag: int
    analysis_start_date: str


# =============================================================================
# DATA ADAPTER CONFIGURATION
# =============================================================================


class DataPaths(TypedDict):
    """Paths for data loading operations.

    Attributes
    ----------
    sales_path : str
        Path to sales data (S3 prefix or local path)
    rates_path : str
        Path to WINK rates data
    weights_path : str
        Path to market weights data
    output_path : str
        Path for outputs
    """

    sales_path: str
    rates_path: str
    weights_path: str
    output_path: str


class DataPathsOptional(TypedDict, total=False):
    """Optional data path fields."""

    macro_path: str
    cache_path: str


# =============================================================================
# AGGREGATION CONFIGURATION
# =============================================================================


class AggregationConfig(TypedDict):
    """Configuration for competitor rate aggregation.

    Attributes
    ----------
    method : str
        Aggregation method: "weighted", "top_n", "firm_level"
    n_competitors : int
        Number of competitors to include (for top_n)
    min_companies : int
        Minimum companies required for valid calculation
    exclude_own : bool
        Whether to exclude own company from competitors
    """

    method: str
    n_competitors: int
    min_companies: int
    exclude_own: bool


# =============================================================================
# CONSTRAINT CONFIGURATION
# =============================================================================


class ConstraintConfig(TypedDict):
    """Configuration for economic constraint validation.

    Attributes
    ----------
    feature_pattern : str
        Regex pattern to match features
    expected_sign : str
        Expected coefficient sign: "positive" or "negative"
    strict : bool
        Whether violation fails (True) or warns (False)
    business_rationale : str
        Explanation for the constraint
    """

    feature_pattern: str
    expected_sign: str
    strict: bool
    business_rationale: str


# =============================================================================
# RESULTS TYPES
# =============================================================================


class InferenceResults(TypedDict):
    """Results from price elasticity inference.

    Attributes
    ----------
    coefficients : Dict[str, float]
        Model coefficients by feature name
    confidence_intervals : Dict[str, Dict[str, Tuple[float, float]]]
        CIs by feature and confidence level
    elasticity_point : float
        Point estimate of price elasticity
    elasticity_ci : Tuple[float, float]
        95% CI for elasticity
    model_fit : Dict[str, float]
        Model fit statistics (R², AIC, etc.)
    n_observations : int
        Number of observations used
    """

    coefficients: Dict[str, float]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    elasticity_point: float
    elasticity_ci: Tuple[float, float]
    model_fit: Dict[str, float]
    n_observations: int


# FeatureSelectionResults has been moved to src.features.selection_types
# Use: from src.features.selection_types import FeatureSelectionResults
# This TypedDict has been deleted in favor of the dataclass for DRY compliance.


__all__ = [
    "AWSConfig",
    "AWSConfigOptional",
    "InferenceConfig",
    "InferenceConfigOptional",
    "FeatureConfig",
    "DataPaths",
    "DataPathsOptional",
    "AggregationConfig",
    "ConstraintConfig",
    "InferenceResults",
    # FeatureSelectionResults removed - use src.features.selection_types
]
