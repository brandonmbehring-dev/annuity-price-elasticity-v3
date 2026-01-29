"""
Type Definitions for Feature Selection System.

This module provides comprehensive type definitions for the feature selection
pipeline, ensuring type safety and clear interfaces across all atomic functions.

Design Principles:
- Complete type coverage for all configuration objects
- Business-oriented parameter names and documentation
- Power-user flexibility with sensible defaults
- Compatibility with existing notebook patterns
"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Required, NotRequired
from dataclasses import dataclass
from enum import Enum

# Canonical feature configuration (Single Source of Truth)
from src.config.product_config import get_default_feature_config, ProductFeatureConfig


class ConstraintType(Enum):
    """Economic constraint types for business rule validation."""
    COMPETITOR_NEGATIVE = "competitor_negative"
    PRUDENTIAL_POSITIVE = "prudential_positive"
    AUTOREGRESSIVE_POSITIVE = "autoregressive_positive"


# Import canonical FeatureSelectionConfig from pipeline_config
from src.config.pipeline_config import FeatureSelectionConfig


class EconomicConstraintConfig(TypedDict):
    """
    Configuration for economic constraint validation.

    Provides full control over business rule validation with
    the ability to define custom constraint rules.
    """
    # Core constraint settings
    enabled: Required[bool]
    strict_validation: NotRequired[bool]  # Fail on any violation vs. warn

    # Constraint rule definitions (power users can customize)
    competitor_rules: NotRequired[Dict[str, Any]]
    prudential_rules: NotRequired[Dict[str, Any]]
    autoregressive_rules: NotRequired[Dict[str, Any]]
    custom_rules: NotRequired[List[Dict[str, Any]]]


class BootstrapAnalysisConfig(TypedDict):
    """
    Configuration for bootstrap stability analysis.

    Full control over bootstrap parameters for power users
    who want to customize uncertainty quantification.
    """
    # Core bootstrap settings
    enabled: Required[bool]
    n_samples: Required[int]
    models_to_analyze: Required[int]

    # Advanced settings for power users
    confidence_intervals: NotRequired[List[int]]  # e.g., [90, 95, 99]
    random_seed: NotRequired[int]
    parallel_execution: NotRequired[bool]

    # Time series bootstrap settings
    use_block_bootstrap: NotRequired[bool]  # True for time series data
    block_size: NotRequired[int]  # Block size for time series bootstrap
    bootstrap_method: NotRequired[str]  # "standard", "block", "stationary"

    # Stability metrics
    calculate_stability_metrics: NotRequired[bool]
    stability_threshold: NotRequired[float]


class ExperimentConfig(TypedDict):
    """
    Configuration for MLflow experiment tracking.

    Complete control over experiment logging for power users
    who want detailed tracking and reproducibility.
    """
    # Core experiment settings
    enabled: Required[bool]
    experiment_name: Required[str]

    # Logging control (power user options)
    log_coefficients: NotRequired[bool]
    log_intermediate_results: NotRequired[bool]
    log_feature_combinations: NotRequired[bool]
    log_constraint_violations: NotRequired[bool]

    # Advanced tracking
    auto_tag_best_model: NotRequired[bool]
    custom_tags: NotRequired[Dict[str, str]]
    artifact_logging: NotRequired[bool]


@dataclass
class AICResult:
    """
    Structured result from model evaluation calculation.

    Comprehensive result object that maintains compatibility
    with existing notebook output while providing structured access.
    Now includes AIC, BIC, R², and adjusted R² for complete model comparison.

    Standard errors are included to support CI-based constraint validation
    (Issue #6 from audit: constraints validated on point estimates only).
    """
    features: str
    n_features: int
    aic: float
    bic: float
    r_squared: float
    r_squared_adj: float
    coefficients: Dict[str, float]
    converged: bool
    n_obs: int
    standard_errors: Optional[Dict[str, float]] = None  # SE for CI-based constraints
    error: Optional[str] = None

    def __post_init__(self):
        """Validate model result consistency."""
        if self.converged and (self.aic == float('inf') or self.bic == float('inf')):
            self.converged = False
            if not self.error:
                self.error = "Model marked as converged but AIC/BIC is infinite"


@dataclass
class ConstraintRule:
    """
    Single economic constraint rule definition.

    Allows power users to define custom business rules
    with clear rationale and validation logic.
    """
    feature_pattern: str
    expected_sign: str  # "positive" or "negative"
    constraint_type: ConstraintType
    business_rationale: str
    strict: bool = True  # Whether violation should fail or warn


@dataclass
class ConstraintViolation:
    """
    Detailed constraint violation information.

    Provides comprehensive information about economic
    constraint violations for debugging and validation.
    """
    feature_name: str
    actual_coefficient: float
    expected_sign: str
    constraint_type: ConstraintType
    business_rationale: str
    violation_severity: str  # "ERROR" or "WARNING"


@dataclass
class BootstrapResult:
    """
    Bootstrap stability analysis result.

    Comprehensive bootstrap analysis with stability metrics
    and detailed uncertainty quantification.
    """
    model_name: str
    model_features: str
    bootstrap_aics: List[float]
    bootstrap_r2_values: List[float]
    original_aic: float
    original_r2: float

    # Stability metrics
    aic_stability_coefficient: float
    r2_stability_coefficient: float
    confidence_intervals: Dict[str, Dict[str, float]]  # {metric: {percentile: value}}

    # Quality indicators
    successful_fits: int
    total_attempts: int
    stability_assessment: str  # "STABLE", "MODERATE", "UNSTABLE"


@dataclass
class FeatureSelectionResults:
    """
    Complete feature selection analysis results.

    Comprehensive result object containing all analysis outputs
    with structured access to all components.

    This is the single source of truth for feature selection results.
    The TypedDict version in core/types.py has been removed in favor of this dataclass.
    """
    # All required fields first (no defaults)
    best_model: AICResult
    all_results: 'pd.DataFrame'  # Use string to avoid circular import
    valid_results: 'pd.DataFrame'
    total_combinations: int
    converged_models: int
    economically_valid_models: int
    constraint_violations: List[ConstraintViolation]
    feature_config: FeatureSelectionConfig
    constraint_config: EconomicConstraintConfig

    # All optional fields with defaults
    bootstrap_results: Optional[List[BootstrapResult]] = None
    stability_summary: Optional[Dict[str, Any]] = None
    bootstrap_config: Optional[BootstrapAnalysisConfig] = None
    experiment_config: Optional[ExperimentConfig] = None
    execution_time_seconds: Optional[float] = None
    mlflow_run_id: Optional[str] = None

    @property
    def selected_features(self) -> List[str]:
        """Get selected features from best model.

        Provides compatibility with TypedDict-style interface.
        """
        if isinstance(self.best_model.features, str):
            # features is stored as "f1 + f2 + f3" string
            return [f.strip() for f in self.best_model.features.split('+')]
        return list(self.best_model.features) if self.best_model.features else []

    @property
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance from best model coefficients.

        Provides compatibility with TypedDict-style interface.
        """
        return self.best_model.coefficients or {}

    @property
    def validation_passed(self) -> bool:
        """Check if economic constraints passed.

        Provides compatibility with TypedDict-style interface.
        """
        return self.economically_valid_models > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility.

        Returns a structure compatible with the old TypedDict interface:
        - selected_features: List[str]
        - feature_importance: Dict[str, float]
        - selection_criteria: Dict[str, Any]
        - validation_passed: bool
        - constraint_violations: List[str]
        """
        return {
            # TypedDict-compatible fields
            "selected_features": self.selected_features,
            "feature_importance": self.feature_importance,
            "selection_criteria": {
                "aic": self.best_model.aic,
                "bic": self.best_model.bic,
                "r_squared": self.best_model.r_squared,
                "n_features": self.best_model.n_features,
            },
            "validation_passed": self.validation_passed,
            "constraint_violations": [str(v) for v in self.constraint_violations],
            # Additional fields from dataclass
            "best_model": str(self.best_model),
            "all_results": str(self.all_results),
            "valid_results": str(self.valid_results),
            "total_combinations": self.total_combinations,
            "converged_models": self.converged_models,
            "economically_valid_models": self.economically_valid_models,
        }


# Default configuration builders for power users
def create_default_constraint_rules() -> List[ConstraintRule]:
    """
    Create default economic constraint rules.

    Power users can use this as a starting point and customize
    by adding, removing, or modifying rules as needed.
    """
    return [
        ConstraintRule(
            feature_pattern="competitor_",
            expected_sign="negative",
            constraint_type=ConstraintType.COMPETITOR_NEGATIVE,
            business_rationale="Higher competitor rates should increase our sales (competitive advantage)",
            strict=True
        ),
        ConstraintRule(
            feature_pattern="prudential_rate",
            expected_sign="positive",
            constraint_type=ConstraintType.PRUDENTIAL_POSITIVE,
            business_rationale="Higher Prudential rates should increase sales (pricing power)",
            strict=True
        ),
        ConstraintRule(
            feature_pattern="sales_target_t",
            expected_sign="positive",
            constraint_type=ConstraintType.AUTOREGRESSIVE_POSITIVE,
            business_rationale="Sales persistence - past sales predict future sales",
            strict=False  # More lenient on autoregressive terms
        ),
        ConstraintRule(
            feature_pattern="sales_target_contract_t",
            expected_sign="positive",
            constraint_type=ConstraintType.AUTOREGRESSIVE_POSITIVE,
            business_rationale="Contract date sales persistence",
            strict=False
        )
    ]


def _get_default_candidate_features() -> List[str]:
    """
    Return default candidate features for feature selection.

    Delegates to ProductFeatureConfig (Single Source of Truth in product_config.py).

    Returns
    -------
    List[str]
        Default candidate feature names
    """
    config = get_default_feature_config()
    return list(config.candidate_features)


def _build_feature_config(
    base_features: List[str],
    candidate_features: List[str],
    max_candidate_features: int,
    target_variable: str,
    analysis_start_date: str,
    target_transformation: str,
    transformed_target_suffix: str
) -> FeatureSelectionConfig:
    """
    Build feature selection configuration.

    Parameters
    ----------
    base_features : List[str]
        Base features always included in models
    candidate_features : List[str]
        Candidate features to evaluate
    max_candidate_features : int
        Maximum number of candidate features to select
    target_variable : str
        Target variable column name
    analysis_start_date : str
        Analysis start date filter
    target_transformation : str
        Target transformation type
    transformed_target_suffix : str
        Custom suffix for transformed target column

    Returns
    -------
    FeatureSelectionConfig
        Configured feature selection settings
    """
    return FeatureSelectionConfig(
        base_features=base_features,
        candidate_features=candidate_features,
        max_candidate_features=max_candidate_features,
        target_variable=target_variable,
        analysis_start_date=analysis_start_date,
        exclude_holidays=True,
        target_transformation=target_transformation,
        transformed_target_suffix=transformed_target_suffix
    )


def build_constraint_config(
    enabled: bool = True,
    strict_validation: bool = True,
    constraint_rules: Optional[Dict[str, bool]] = None,
    violation_tolerance: float = 0.0,
    business_validation: bool = True,
) -> EconomicConstraintConfig:
    """
    Build economic constraint configuration with customizable parameters.

    CANONICAL BUILDER: Use this function for all constraint configuration needs.

    Parameters
    ----------
    enabled : bool, default=True
        Enable constraint validation
    strict_validation : bool, default=True
        Fail on any violation vs. warn only
    constraint_rules : Dict[str, bool], optional
        Custom constraint rules (e.g., {'competitor_negative': True})
    violation_tolerance : float, default=0.0
        Tolerance for constraint violations (0.0 = zero tolerance)
    business_validation : bool, default=True
        Enable business rule validation

    Returns
    -------
    EconomicConstraintConfig
        Configured constraint settings
    """
    config: EconomicConstraintConfig = {
        'enabled': enabled,
        'strict_validation': strict_validation,
    }

    # Add optional constraint rules if provided
    if constraint_rules:
        # Map flat rules to TypedDict structure
        if 'competitor_negative' in constraint_rules:
            config['competitor_rules'] = {'require_negative': constraint_rules['competitor_negative']}
        if 'prudential_positive' in constraint_rules:
            config['prudential_rules'] = {'require_positive': constraint_rules['prudential_positive']}
        if 'autoregressive_positive' in constraint_rules:
            config['autoregressive_rules'] = {'require_positive': constraint_rules['autoregressive_positive']}

    return config


def _build_constraint_config() -> EconomicConstraintConfig:
    """
    Build default economic constraint configuration.

    Note: Prefer using build_constraint_config() for new code.

    Returns
    -------
    EconomicConstraintConfig
        Default constraint settings with strict validation enabled
    """
    return build_constraint_config()


def build_bootstrap_config(
    enabled: bool = True,
    n_samples: int = 100,
    models_to_analyze: int = 10,
    confidence_intervals: Optional[List[int]] = None,
    random_seed: Optional[int] = None,
    use_block_bootstrap: bool = True,
    block_size: int = 10,
    bootstrap_method: str = "block",
    calculate_stability_metrics: bool = True,
    stability_threshold: float = 0.2,
) -> BootstrapAnalysisConfig:
    """
    Build bootstrap analysis configuration with customizable parameters.

    CANONICAL BUILDER: Use this function for all bootstrap configuration needs.

    Parameters
    ----------
    enabled : bool, default=True
        Enable bootstrap analysis
    n_samples : int, default=100
        Number of bootstrap samples
    models_to_analyze : int, default=10
        Number of top models to analyze for stability
    confidence_intervals : List[int], optional
        Confidence interval percentiles (default: [50, 70, 90])
    random_seed : int, optional
        Random seed for reproducibility
    use_block_bootstrap : bool, default=True
        Use block bootstrap for time series data
    block_size : int, default=10
        Block size for time series bootstrap
    bootstrap_method : str, default="block"
        Bootstrap method ("standard", "block", "stationary")
    calculate_stability_metrics : bool, default=True
        Calculate stability metrics
    stability_threshold : float, default=0.2
        Threshold for stability classification

    Returns
    -------
    BootstrapAnalysisConfig
        Configured bootstrap analysis settings
    """
    config: BootstrapAnalysisConfig = {
        'enabled': enabled,
        'n_samples': n_samples,
        'models_to_analyze': models_to_analyze,
        'confidence_intervals': confidence_intervals or [50, 70, 90],
        'use_block_bootstrap': use_block_bootstrap,
        'block_size': block_size,
        'bootstrap_method': bootstrap_method,
        'calculate_stability_metrics': calculate_stability_metrics,
        'stability_threshold': stability_threshold,
    }

    if random_seed is not None:
        config['random_seed'] = random_seed

    return config


def _build_bootstrap_config() -> BootstrapAnalysisConfig:
    """
    Build default bootstrap analysis configuration.

    Note: Prefer using build_bootstrap_config() for new code.

    Returns
    -------
    BootstrapAnalysisConfig
        Default bootstrap analysis settings
    """
    return build_bootstrap_config()


def _build_experiment_config() -> ExperimentConfig:
    """
    Build default MLflow experiment configuration.

    Returns
    -------
    ExperimentConfig
        Default experiment tracking settings
    """
    return ExperimentConfig(
        enabled=True,
        experiment_name="rila_feature_selection",
        log_coefficients=True,
        log_intermediate_results=True,
        auto_tag_best_model=True,
        artifact_logging=True
    )


def build_feature_selection_configs(
    max_candidate_features: int = 4,
    target_variable: str = "sales_target_current",
    analysis_start_date: str = "2022-08-01",
    candidate_features: Optional[List[str]] = None,
    base_features: Optional[List[str]] = None,
    target_transformation: str = "none",
    transformed_target_suffix: str = ""
) -> Dict[str, Any]:
    """
    Build comprehensive feature selection configuration objects.

    DRY principle implementation: consolidates all configuration creation
    patterns from notebook into a single, reusable builder function.

    This eliminates duplicate parameter definitions and provides
    type-safe configuration management following established patterns.

    Parameters
    ----------
    max_candidate_features : int, default 4
        Maximum number of candidate features to select
    target_variable : str, default "sales_target_current"
        Target variable column name
    analysis_start_date : str, default "2022-08-01"
        Analysis start date filter
    candidate_features : List[str], optional
        Custom candidate features list. If None, uses default selection.
    base_features : List[str], optional
        Base features always included. If None, uses empty list.
    target_transformation : str, default "none"
        Target transformation type:
        - "none": No transformation (original target)
        - "log1p": np.log(1 + target) - recommended for count/monetary data
        - "log": np.log(target) - requires positive values only
        - "sqrt": np.sqrt(target) - square root transformation
    transformed_target_suffix : str, default ""
        Custom suffix for transformed target column name

    Returns
    -------
    Dict[str, Any]
        Complete configuration dictionary with all components:
        - feature_config: FeatureSelectionConfig
        - constraint_config: EconomicConstraintConfig
        - bootstrap_config: BootstrapAnalysisConfig
        - experiment_config: ExperimentConfig
    """
    # Apply defaults for None parameters
    resolved_candidates = candidate_features if candidate_features is not None else _get_default_candidate_features()
    resolved_base = base_features if base_features is not None else []

    # Build each configuration component using helpers
    feature_config = _build_feature_config(
        base_features=resolved_base,
        candidate_features=resolved_candidates,
        max_candidate_features=max_candidate_features,
        target_variable=target_variable,
        analysis_start_date=analysis_start_date,
        target_transformation=target_transformation,
        transformed_target_suffix=transformed_target_suffix
    )

    return {
        'feature_config': feature_config,
        'constraint_config': _build_constraint_config(),
        'bootstrap_config': _build_bootstrap_config(),
        'experiment_config': _build_experiment_config()
    }