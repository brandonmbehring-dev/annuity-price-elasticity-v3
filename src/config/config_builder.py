"""
CANONICAL Configuration Builder - THE ONLY CONFIG PATTERN TO USE

This is the authoritative configuration module used by ALL refactored notebooks.
For ANY configuration needs, ALWAYS start here.

CANONICAL FUNCTIONS:
- build_pipeline_configs(): Main function for pipeline configuration
- build_pipeline_configs_for_product(): Product-aware pipeline configuration (recommended)
- build_feature_selection_stage_config(): Feature selection configuration
- build_forecasting_stage_config(): Forecasting configuration

SINGLE SOURCE OF TRUTH:
- No duplicate config classes (all consolidated)
- Clear separation: pipeline_config.py vs forecasting_config.py
- No "enhanced" config variants
- ProductConfig for multi-product support (see src/config/product_config.py)

ARCHITECTURE (Phase 2.3 Refactoring):
- pipeline_builders.py: Pipeline stage configurations
- forecasting_builders.py: Forecasting stage configurations
- inference_builders.py: Inference/metadata configurations
- visualization_builders.py: Visualization configurations
- config_builder.py: Feature selection + backward-compatible re-exports

Usage Pattern (from refactored notebooks):
    # Product-aware approach (recommended)
    from src.config.config_builder import build_pipeline_configs_for_product
    configs = build_pipeline_configs_for_product("6Y20B")
    buffer = configs['product'].buffer_level  # 0.20

    # Legacy approach (still supported)
    from src.config.config_builder import build_pipeline_configs
    configs = build_pipeline_configs(version=6, product_name="FlexGuard indexed variable annuity")
"""

from typing import List

# =============================================================================
# CONTEXT ANCHOR: CONFIGURATION BUILDER OBJECTIVES
# =============================================================================
# PURPOSE: Single source of truth for ALL configuration needs in the entire pipeline
# USED BY: All refactored notebooks (00_, 01_, 02_), no alternative config systems
# DEPENDENCIES: pipeline_config.py (TypedDict definitions), forecasting_config.py (forecasting types),
#               product_config.py (multi-product support)
# LAST VALIDATED: 2026-01-19 (Phase 2.3 - Builder module extraction)
# PATTERN STATUS: CANONICAL (no competing implementations exist)
#
# ARCHITECTURAL FLOW: business parameters -> config_builder -> TypedDict configs -> pipeline functions
# SUCCESS CRITERIA: All configurations built through these functions, zero duplicate config creation
# INTEGRATION: Works with all refactored notebooks without import errors
# MAINTENANCE: Run pattern_validator.py to ensure no competing config systems reintroduced

# Multi-product configuration support (Single Source of Truth for features)
from src.config.types.product_config import (
    ProductConfig, PRODUCT_REGISTRY, get_product_config, get_default_product,
    get_pipeline_product_ids_as_lists, get_metadata_product_ids_as_lists,
    ProductFeatureConfig, get_default_feature_config
)

from src.config.types.pipeline_config import (
    FeatureSelectionConfig, AICAnalysisConfig,
    FeatureValidationConfig, ModelSelectionConfig, FeatureSelectionStageConfig,
)

# =============================================================================
# RE-EXPORTS FOR BACKWARD COMPATIBILITY
# =============================================================================
# These imports ensure existing code continues to work without modification.
# All canonical implementations now live in dedicated builder modules.

from src.config.builders.pipeline_builders import (
    get_lag_column_configs,
    get_weekly_aggregation_dict,
    build_pipeline_configs,
    build_pipeline_configs_for_product,
    _get_default_flexguard_product_ids,  # Deprecated: use get_pipeline_product_ids_as_lists()
)

from src.config.builders.forecasting_builders import (
    build_forecasting_stage_config,
)

from src.config.builders.inference_builders import (
    build_inference_config,
    build_inference_stage_config,
    build_product_metadata_config,
    build_rate_scenario_config,
    build_confidence_interval_config,
    build_tableau_formatting_config,
    _get_default_metadata_product_ids,  # Deprecated: use get_metadata_product_ids_as_lists()
)

from src.config.builders.visualization_builders import (
    build_visualization_config,
)


# =============================================================================
# FEATURE SELECTION CONFIGURATION BUILDERS
# =============================================================================
# These remain in config_builder.py as they are already well-organized and
# follow the 30-50 line function guideline.


def _get_default_base_features() -> List[str]:
    """Return default base features for AIC feature selection.

    Feature Naming Unification (2026-01-26): Uses _t0 naming.

    Returns
    -------
    List[str]
        Default base features (always included in models)
    """
    return ["prudential_rate_t0"]


def _get_default_competitor_candidates() -> List[str]:
    """Return default competitor candidate features for AIC selection.

    Derives from ProductFeatureConfig (Single Source of Truth), filtering
    to competitor-only features.

    Returns
    -------
    List[str]
        Competitor candidate features for selection
    """
    config = get_default_feature_config()
    return [f for f in config.candidate_features if 'competitor' in f.lower()]


def build_aic_feature_selection_config(
    target_variable: str = "sales_target_t0",
    base_features: List[str] = None,
    competitor_candidate_features: List[str] = None,
    max_candidate_features: int = 3,
    economic_constraints: bool = True,
    min_observations: int = 30,
    validation_split: float = 0.3
) -> FeatureSelectionConfig:
    """Build AIC-based feature selection configuration for RILA analysis.

    Feature Naming Unification (2026-01-26): Uses _t0 naming.

    Parameters
    ----------
    target_variable : str
        Name of the dependent variable (default: "sales_target_t0")
    base_features : List[str], optional
        Features that must be included in all models
    competitor_candidate_features : List[str], optional
        Competitor features to evaluate for selection
    max_candidate_features : int
        Maximum number of candidate features to select (default: 3)
    economic_constraints : bool
        Whether to apply economic theory constraints (default: True)
    min_observations : int
        Minimum observations required for AIC analysis (default: 30)
    validation_split : float
        Fraction of data for validation (default: 0.3)

    Returns
    -------
    FeatureSelectionConfig
        Complete feature selection configuration
    """
    base = base_features if base_features is not None else _get_default_base_features()
    candidates = competitor_candidate_features if competitor_candidate_features is not None else _get_default_competitor_candidates()

    return FeatureSelectionConfig({
        'target_variable': target_variable,
        'base_features': base,
        'candidate_features': candidates,
        'max_candidate_features': max_candidate_features,
        'economic_constraints': economic_constraints,
        'min_observations': min_observations,
        'validation_split': validation_split
    })


def build_aic_analysis_config(
    max_models_to_test: int = 1000,
    convergence_tolerance: float = 1e-6,
    r_squared_threshold: float = 0.5,
    p_value_threshold: float = 0.05
) -> AICAnalysisConfig:
    """Build AIC analysis configuration with model selection criteria.

    Parameters
    ----------
    max_models_to_test : int
        Maximum number of model combinations to evaluate (default: 1000)
    convergence_tolerance : float
        Numerical convergence tolerance (default: 1e-6)
    r_squared_threshold : float
        Minimum R-squared threshold for model acceptance (default: 0.5)
    p_value_threshold : float
        Maximum p-value threshold for significance (default: 0.05)

    Returns
    -------
    AICAnalysisConfig
        Complete AIC analysis configuration

    Examples
    --------
    >>> config = build_aic_analysis_config(max_models_to_test=500)
    >>> assert config['max_models_to_test'] == 500
    >>> assert config['convergence_tolerance'] == 1e-6
    """
    return AICAnalysisConfig({
        'max_models_to_test': max_models_to_test,
        'convergence_tolerance': convergence_tolerance,
        'r_squared_threshold': r_squared_threshold,
        'p_value_threshold': p_value_threshold,
        'coefficient_constraints': {
            'competitor_': 'negative',
            'prudential_rate': 'positive'
        }
    })


def build_feature_validation_config(
    require_competitor_features: bool = True,
    require_prudential_features: bool = True,
    max_correlation_threshold: float = 0.9,
    cross_validation_folds: int = 5
) -> FeatureValidationConfig:
    """Build feature validation configuration for constraint checking.

    Parameters
    ----------
    require_competitor_features : bool
        Whether competitor features are required (default: True)
    require_prudential_features : bool
        Whether Prudential features are required (default: True)
    max_correlation_threshold : float
        Maximum correlation between features (default: 0.9)
    cross_validation_folds : int
        Number of CV folds for validation (default: 5)

    Returns
    -------
    FeatureValidationConfig
        Complete feature validation configuration

    Examples
    --------
    >>> config = build_feature_validation_config(cross_validation_folds=10)
    >>> assert config['cross_validation_folds'] == 10
    >>> assert config['require_competitor_features'] == True
    """
    return FeatureValidationConfig({
        'require_competitor_features': require_competitor_features,
        'require_prudential_features': require_prudential_features,
        'max_correlation_threshold': max_correlation_threshold,
        'min_feature_importance': 0.01,
        'cross_validation_folds': cross_validation_folds
    })


def build_model_selection_config(
    selection_criteria: str = 'aic',
    ensemble_size: int = 5,
    bootstrap_iterations: int = 100,
    confidence_level: float = 0.95
) -> ModelSelectionConfig:
    """Build model selection configuration for ensemble and ranking.

    Parameters
    ----------
    selection_criteria : str
        Primary selection criteria (default: 'aic')
    ensemble_size : int
        Number of models in ensemble (default: 5)
    bootstrap_iterations : int
        Bootstrap iterations for stability (default: 100)
    confidence_level : float
        Confidence level for intervals (default: 0.95)

    Returns
    -------
    ModelSelectionConfig
        Complete model selection configuration

    Examples
    --------
    >>> config = build_model_selection_config(ensemble_size=3)
    >>> assert config['ensemble_size'] == 3
    >>> assert config['selection_criteria'] == 'aic'
    """
    return ModelSelectionConfig({
        'selection_criteria': selection_criteria,
        'ensemble_size': ensemble_size,
        'bootstrap_iterations': bootstrap_iterations,
        'confidence_level': confidence_level,
        'performance_metrics': ['aic', 'r_squared', 'mape', 'rmse']
    })


def build_feature_selection_stage_config(
    target_variable: str = "sales_target_current",
    base_features: List[str] = None,
    max_candidate_features: int = 3,
    economic_constraints: bool = True
) -> FeatureSelectionStageConfig:
    """Build complete feature selection stage configuration combining all sub-configs.

    Parameters
    ----------
    target_variable : str
        Target variable for analysis (default: "sales_target_current")
    base_features : List[str], optional
        Required base features for all models
    max_candidate_features : int
        Maximum candidate features to select (default: 3)
    economic_constraints : bool
        Whether to apply economic theory constraints (default: True)

    Returns
    -------
    FeatureSelectionStageConfig
        Complete feature selection stage configuration

    Examples
    --------
    >>> config = build_feature_selection_stage_config(max_candidate_features=4)
    >>> assert 'selection_config' in config
    >>> assert 'analysis_config' in config
    >>> assert config['selection_config']['max_candidate_features'] == 4
    """
    return FeatureSelectionStageConfig({
        'selection_config': build_aic_feature_selection_config(
            target_variable=target_variable,
            base_features=base_features,
            max_candidate_features=max_candidate_features,
            economic_constraints=economic_constraints
        ),
        'analysis_config': build_aic_analysis_config(),
        'validation_config': build_feature_validation_config(),
        'model_selection_config': build_model_selection_config()
    })
