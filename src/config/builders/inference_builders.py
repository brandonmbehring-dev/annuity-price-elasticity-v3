"""
Inference Stage Configuration Builders

Extracted from config_builder.py for single-responsibility compliance.
Handles all inference stage configurations: core inference, rate scenarios,
confidence intervals, Tableau formatting, and product metadata.

Usage:
    from src.config.inference_builders import build_inference_stage_config
    config = build_inference_stage_config(n_estimators=10000)
"""

from typing import Dict, List, Optional, Any

from src.config.types.product_config import get_metadata_product_ids_as_lists
from src.config.types.forecasting_config import (
    InferenceConfig, RateScenarioConfig, ConfidenceIntervalConfig,
    TableauFormattingConfig, ProductMetadataConfig, InferenceStageConfig
)


# Module exports (including private functions for testing)
__all__ = [
    # Public API
    'build_inference_config',
    'build_inference_stage_config',
    'build_product_metadata_config',
    'build_rate_scenario_config',
    'build_confidence_interval_config',
    'build_tableau_formatting_config',
    # Private helpers (for unit testing)
    '_get_default_inference_features',
    '_get_default_metadata_product_ids',
    '_get_default_product_name_dict',
    '_build_core_inference_configs',
    '_build_output_configs',
]


# =============================================================================
# INFERENCE CONFIGURATION HELPERS
# =============================================================================


def _get_default_inference_features() -> List[str]:
    """Return default features for inference configuration.

    Returns
    -------
    List[str]
        Default feature names for inference models
    """
    return [
        'competitor_mid_t2',
        'competitor_top5_t2',
        'prudential_rate_current',
        'prudential_rate_t3'
    ]


def build_inference_config(
    n_estimators: int = 10000,
    weight_decay_factor: float = 0.99,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
    sales_multiplier: float = 13.0,
    momentum_lookback_periods: int = 3,
    training_cutoff_days: int = 60,
    target_column: str = "sales_target_current",
    features: Optional[List[str]] = None,
    exclude_zero_sales: bool = True,
    date_filter_start: str = "2022-04-01"
) -> InferenceConfig:
    """Build inference configuration for RILA price elasticity analysis.

    Parameters
    ----------
    n_estimators : int, default=10000
        Number of bootstrap estimators
    weight_decay_factor : float, default=0.99
        Decay factor for time weighting
    random_state : int, default=42
        Random seed for reproducibility
    ridge_alpha : float, default=1.0
        Ridge regularization parameter
    sales_multiplier : float, default=13.0
        Sales scaling factor
    momentum_lookback_periods : int, default=3
        Periods for momentum calculation
    training_cutoff_days : int, default=60
        Days before end for training cutoff
    target_column : str, default="sales_target_current"
        Target variable column name (STANDARDIZED key)
    features : Optional[List[str]]
        Feature columns for model
    exclude_zero_sales : bool, default=True
        Whether to exclude zero-sale rows
    date_filter_start : str, default="2022-04-01"
        Start date for data filtering

    Returns
    -------
    InferenceConfig
        Complete inference configuration
    """
    resolved_features = features if features is not None else _get_default_inference_features()

    return InferenceConfig({
        'n_estimators': n_estimators,
        'weight_decay_factor': weight_decay_factor,
        'random_state': random_state,
        'ridge_alpha': ridge_alpha,
        'sales_multiplier': sales_multiplier,
        'momentum_lookback_periods': momentum_lookback_periods,
        'training_cutoff_days': training_cutoff_days,
        'target_column': target_column,
        # Backward compatibility shim: also include target_variable
        'target_variable': target_column,
        'features': resolved_features,
        'exclude_zero_sales': exclude_zero_sales,
        'date_filter_start': date_filter_start
    })


# =============================================================================
# PRODUCT METADATA CONFIGURATION
# =============================================================================


def _get_default_metadata_product_ids() -> Dict[str, List[int]]:
    """Return default FlexGuard product ID mapping for metadata configuration.

    DEPRECATED: Use get_metadata_product_ids_as_lists() from product_config
    for the canonical source. This function is maintained for backward
    compatibility and delegates to the canonical source.

    Note: This differs from _get_default_flexguard_product_ids() used in pipeline config.
    Metadata config includes additional product variants for complete business reporting.

    Returns
    -------
    Dict[str, List[int]]
        Metadata product ID mapping (historical data)
    """
    return get_metadata_product_ids_as_lists()


def _get_default_product_name_dict() -> Dict[str, str]:
    """Return default company to full product name mapping.

    Returns
    -------
    Dict[str, str]
        Company code to full product name mapping
    """
    return {
        "Prudential": "Prudential Flexguard Indexed Variable Annuity",
        "Allianz": "Allianz Index Advantage+ NF Variable Annuity",
        "Athene": "Athene Amplify 2.0 NF",
        "Brighthouse": "Brighthouse Shield Level Select 6-year",
        "Equitable": "Structured Capital Strategies Plus 21",
        "Jackson": "Jackson Market Link Pro Advisory Single Premium Deferred Index-Linked Annuity",
        "Lincoln": "Lincoln Level Advantage Design Advisory Share Individual Variable and Indexed-Linked Annuity",
        "Symetra": "Symetra Trek Frontier Index-Linked Annuity",
        "Trans": "Transamerica Structured Index Advantage Annuity",
        "weighted_mean": "Weighted Mean By Market Share of Competitors Cap Rate",
    }


def build_product_metadata_config(
    product_name: str = "FlexGuard_6Y20B",
    version: str = "v2_1",
    flexguard_product_ids: Dict[str, List[int]] = None,
    product_name_dict: Optional[Dict[str, str]] = None
) -> ProductMetadataConfig:
    """Build product metadata configuration with business mappings.

    Parameters
    ----------
    product_name : str, default="FlexGuard_6Y20B"
        Product identifier for output metadata
    version : str, default="v2_1"
        Version identifier for export files
    flexguard_product_ids : Dict[str, List[int]], optional
        Company to FlexGuard product ID mapping
    product_name_dict : Dict[str, str], optional
        Company to full product name mapping

    Returns
    -------
    ProductMetadataConfig
        Complete product metadata configuration
    """
    resolved_ids = flexguard_product_ids if flexguard_product_ids is not None else _get_default_metadata_product_ids()
    resolved_names = product_name_dict if product_name_dict is not None else _get_default_product_name_dict()

    return ProductMetadataConfig({
        'product_name': product_name,
        'version': version,
        'flexguard_product_ids': resolved_ids,
        'product_name_dict': resolved_names
    })


# =============================================================================
# RATE SCENARIO AND CONFIDENCE INTERVAL CONFIGURATIONS
# =============================================================================


def build_rate_scenario_config(
    rate_min: float = 0.005,
    rate_max: float = 4.5,
    rate_steps: int = 19,
    competitor_rate_adjustment: float = 0.0
) -> RateScenarioConfig:
    """Build rate scenario configuration for price elasticity analysis.

    Creates rate scenario configuration matching original notebook:
    rate_options = np.linspace(0.005, 4.50, 19)  # Starting at 50bps
    competitor_rate_adjustment = 0

    Parameters
    ----------
    rate_min : float, default=0.005
        Minimum rate for scenario analysis (50 basis points)
    rate_max : float, default=4.5
        Maximum rate for scenario analysis
    rate_steps : int, default=19
        Number of rate scenarios to generate
    competitor_rate_adjustment : float, default=0.0
        Competitor rate adjustment for scenarios

    Returns
    -------
    RateScenarioConfig
        Complete rate scenario configuration

    Examples
    --------
    >>> config = build_rate_scenario_config()
    >>> import numpy as np
    >>> rate_options = np.linspace(config['rate_min'], config['rate_max'], config['rate_steps'])
    >>> assert len(rate_options) == 19
    >>> assert rate_options[0] == 0.005
    >>> assert rate_options[-1] == 4.5
    """
    return RateScenarioConfig({
        'rate_min': rate_min,
        'rate_max': rate_max,
        'rate_steps': rate_steps,
        'competitor_rate_adjustment': competitor_rate_adjustment
    })


def build_confidence_interval_config(
    confidence_level: float = 0.95,
    rounding_precision: int = 3,
    basis_points_multiplier: int = 100
) -> ConfidenceIntervalConfig:
    """Build confidence interval configuration for bootstrap results.

    Parameters
    ----------
    confidence_level : float, default=0.95
        Confidence level for statistical intervals
    rounding_precision : int, default=3
        Decimal precision for rounding results
    basis_points_multiplier : int, default=100
        Multiplier to convert to basis points

    Returns
    -------
    ConfidenceIntervalConfig
        Complete confidence interval configuration

    Examples
    --------
    >>> config = build_confidence_interval_config()
    >>> assert config['confidence_level'] == 0.95
    >>> assert config['rounding_precision'] == 3
    >>> assert config['basis_points_multiplier'] == 100
    """
    return ConfidenceIntervalConfig({
        'confidence_level': confidence_level,
        'rounding_precision': rounding_precision,
        'basis_points_multiplier': basis_points_multiplier
    })


def build_tableau_formatting_config(
    prudential_rate_col: str = "prudential_rate_current",
    competitor_rate_col: str = "competitor_mid_current",
    sales_lag_cols: Optional[List[str]] = None,
    sales_rounding_power: int = -7
) -> TableauFormattingConfig:
    """Build Tableau formatting configuration for output preparation.

    Parameters
    ----------
    prudential_rate_col : str, default="prudential_rate_current"
        Column name for Prudential rate data
    competitor_rate_col : str, default="competitor_mid_current"
        Column name for competitor rate data
    sales_lag_cols : List[str], optional
        Column names for sales lag features
    sales_rounding_power : int, default=-7
        Power of 10 for sales rounding (default rounds to 10M)

    Returns
    -------
    TableauFormattingConfig
        Complete Tableau formatting configuration

    Examples
    --------
    >>> config = build_tableau_formatting_config()
    >>> assert config['prudential_rate_col'] == "prudential_rate_current"
    >>> assert config['sales_rounding_power'] == -7
    """
    if sales_lag_cols is None:
        sales_lag_cols = ['sales_target_t2', 'sales_target_t3']

    return TableauFormattingConfig({
        'prudential_rate_col': prudential_rate_col,
        'competitor_rate_col': competitor_rate_col,
        'sales_lag_cols': sales_lag_cols,
        'sales_rounding_power': sales_rounding_power
    })


# =============================================================================
# INFERENCE STAGE HELPER FUNCTIONS
# =============================================================================
# Note: Trivial wrapper functions removed in TD-08 cleanup (2026-01-24)
# Direct calls to build_inference_config, build_rate_scenario_config,
# build_confidence_interval_config, and build_tableau_formatting_config
# are used instead.


# =============================================================================
# STAGE-BASED INFERENCE HELPERS
# =============================================================================


def _build_core_inference_configs(
    n_estimators: int,
    weight_decay_factor: float,
    random_state: int,
    ridge_alpha: float,
    sales_multiplier: float,
    momentum_lookback_periods: int,
    rate_min: float,
    rate_max: float,
    rate_steps: int,
    competitor_rate_adjustment: float
) -> Dict[str, Any]:
    """Build core inference and rate scenario configs."""
    return {
        'inference_config': build_inference_config(
            n_estimators=n_estimators,
            weight_decay_factor=weight_decay_factor,
            random_state=random_state,
            ridge_alpha=ridge_alpha,
            sales_multiplier=sales_multiplier,
            momentum_lookback_periods=momentum_lookback_periods
        ),
        'rate_scenario_config': build_rate_scenario_config(
            rate_min=rate_min,
            rate_max=rate_max,
            rate_steps=rate_steps,
            competitor_rate_adjustment=competitor_rate_adjustment
        )
    }


def _build_output_configs(
    confidence_level: float,
    rounding_precision: int,
    basis_points_multiplier: int,
    prudential_rate_col: str,
    competitor_rate_col: str,
    sales_lag_cols: Optional[List[str]],
    output_directory: str
) -> Dict[str, Any]:
    """Build output formatting and metadata configs.

    Parameters
    ----------
    confidence_level : float
        Confidence level for intervals
    rounding_precision : int
        Decimal precision
    basis_points_multiplier : int
        Basis points multiplier
    prudential_rate_col : str
        Prudential rate column name
    competitor_rate_col : str
        Competitor rate column name
    sales_lag_cols : Optional[List[str]]
        Sales lag column names
    output_directory : str
        Output directory for visualizations and BI exports

    Returns
    -------
    Dict[str, Any]
        Output configuration dict
    """
    # Import here to avoid circular imports
    from src.config.builders.visualization_builders import build_visualization_config

    return {
        'confidence_interval_config': build_confidence_interval_config(
            confidence_level=confidence_level,
            rounding_precision=rounding_precision,
            basis_points_multiplier=basis_points_multiplier
        ),
        'tableau_formatting_config': build_tableau_formatting_config(
            prudential_rate_col=prudential_rate_col,
            competitor_rate_col=competitor_rate_col,
            sales_lag_cols=sales_lag_cols
        ),
        'product_metadata_config': build_product_metadata_config(),
        'visualization_config': build_visualization_config(output_directory=output_directory)
    }


# =============================================================================
# MAIN INFERENCE STAGE CONFIG BUILDER
# =============================================================================


def build_inference_stage_config(
    n_estimators: int = 1000,
    weight_decay_factor: float = 0.99,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
    sales_multiplier: float = 13.0,
    momentum_lookback_periods: int = 3,
    rate_min: float = 0.005,
    rate_max: float = 4.5,
    rate_steps: int = 19,
    competitor_rate_adjustment: float = 0.0,
    confidence_level: float = 0.95,
    rounding_precision: int = 3,
    basis_points_multiplier: int = 100,
    prudential_rate_col: str = "prudential_rate_current",
    competitor_rate_col: str = "competitor_mid_current",
    sales_lag_cols: Optional[List[str]] = None,
    output_directory: str = "../../outputs/rila_6y20b/bi_team"
) -> InferenceStageConfig:
    """Build complete inference stage configuration.

    Combines inference core, rate scenarios, confidence intervals,
    tableau formatting, product metadata, and visualization configs.

    Parameters
    ----------
    output_directory : str, default="../../outputs/rila_6y20b/bi_team"
        Output directory for visualizations and BI exports.
        Default is relative to notebooks/production/rila_6y20b/.
        For 1Y10B, use "../../outputs/rila_1y10b/bi_team"

    Returns
    -------
    InferenceStageConfig
        Complete inference stage configuration
    """
    config_dict: Dict[str, Any] = {}

    # Core inference and rate scenario configs
    config_dict.update(_build_core_inference_configs(
        n_estimators, weight_decay_factor, random_state, ridge_alpha,
        sales_multiplier, momentum_lookback_periods, rate_min, rate_max,
        rate_steps, competitor_rate_adjustment
    ))

    # Output formatting and metadata configs
    config_dict.update(_build_output_configs(
        confidence_level, rounding_precision, basis_points_multiplier,
        prudential_rate_col, competitor_rate_col, sales_lag_cols,
        output_directory
    ))

    return InferenceStageConfig(config_dict)
