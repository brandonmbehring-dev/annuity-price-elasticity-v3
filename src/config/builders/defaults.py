"""
Centralized Default Values for Configuration Builders.

This module provides a single source of truth for default configuration
values used across all builder modules.

Usage:
    from src.config.builders.defaults import (
        DEFAULTS,
        get_default,
        get_product_defaults,
    )

Design Principles:
- Single Source of Truth: All defaults defined here
- Immutable Defaults: Values are frozen, not modified at runtime
- Override Support: Easy to override individual defaults
"""

from typing import Any, Dict, List, Optional, FrozenSet
from dataclasses import dataclass, field


# =============================================================================
# CORE STATISTICAL DEFAULTS
# =============================================================================

BOOTSTRAP_DEFAULTS = {
    'n_samples': 1000,
    'random_state': 42,
    'confidence_level': 0.95,
    'models_to_analyze': 15,
    'enabled': False,
    'parallel_execution': False,
    'block_size': 4,  # For block bootstrap (4 weeks)
    'use_block_bootstrap': False,
}

CROSS_VALIDATION_DEFAULTS = {
    'start_cutoff': 30,
    'end_cutoff': None,  # Use full dataset
    'validation_method': 'expanding_window',
    'n_splits': 0,  # Continuous expanding window
}

RIDGE_DEFAULTS = {
    'alpha': 1.0,
    'fit_intercept': True,
    'normalize': False,
    'positive_constraint': True,  # Business constraint: sales >= 0
}


# =============================================================================
# FEATURE DEFAULTS
# =============================================================================
# Feature Naming Unification (2026-01-26):
# - All temporal suffixes use _t{N} format: _t0, _t1, _t2, etc.
# - Previous _current suffix normalized to _t0
# - competitor_mid renamed to competitor_weighted for semantic clarity

COMPETITIVE_FEATURES = (
    'competitor_weighted_t2',
    'competitor_top5_t2',
    'competitor_weighted_t3',
    'competitor_top5_t3',
)

INFERENCE_FEATURES = (
    'competitor_weighted_t2',
    'competitor_top5_t2',
    'prudential_rate_t0',
    'prudential_rate_t3',
)

MODEL_FEATURES = (
    'prudential_rate_t0',
    'competitor_weighted_t2',
    'competitor_top5_t3',
)

BENCHMARK_FEATURES = (
    'sales_target_contract_t5',
)

BASE_FEATURES = (
    'prudential_rate_t0',
)


# =============================================================================
# BUSINESS RULE DEFAULTS
# =============================================================================

BUSINESS_FILTER_DEFAULTS = {
    'analysis_start_date': '2022-04-01',
    'temporal_weight_decay_rate': 0.98,
    'remove_incomplete_final_obs': True,
    'mature_data_cutoff_days': 50,
}

ECONOMIC_CONSTRAINT_DEFAULTS = {
    'enabled': True,
    'strict_validation': True,
    'competitor_sign': 'negative',
    'own_rate_sign': 'positive',
}


# =============================================================================
# INFERENCE DEFAULTS
# =============================================================================

INFERENCE_DEFAULTS = {
    'n_estimators': 10000,
    'weight_decay_factor': 0.99,
    'random_state': 42,
    'ridge_alpha': 1.0,
    'sales_multiplier': 13.0,
    'momentum_lookback_periods': 3,
    'training_cutoff_days': 60,
    'target_variable': 'sales_target_t0',  # Unified naming: _current → _t0
    'exclude_zero_sales': True,
    'date_filter_start': '2022-04-01',
}

RATE_SCENARIO_DEFAULTS = {
    'rate_min': 0.005,  # 50 basis points
    'rate_max': 4.5,
    'rate_steps': 19,
    'competitor_rate_adjustment': 0.0,
}

CONFIDENCE_INTERVAL_DEFAULTS = {
    'confidence_level': 0.95,
    'rounding_precision': 3,
    'basis_points_multiplier': 100,
}

TABLEAU_FORMATTING_DEFAULTS = {
    'prudential_rate_col': 'prudential_rate_t0',  # Unified naming: _current → _t0
    'competitor_rate_col': 'competitor_weighted_t0',  # competitor_mid → competitor_weighted
    'sales_lag_cols': ('sales_target_t2', 'sales_target_t3'),
    'sales_rounding_power': -7,  # Rounds to 10M
}


# =============================================================================
# VISUALIZATION DEFAULTS
# =============================================================================

VISUALIZATION_DEFAULTS = {
    'fig_width': 10,
    'fig_height': 8,
    'style': 'whitegrid',
    'palette': 'deep',
    'dpi': 100,
    'save_format': 'png',
}


# =============================================================================
# VALIDATION DEFAULTS
# =============================================================================

VALIDATION_FRAMEWORK_DEFAULTS = {
    'tolerance_r2': 1e-6,
    'tolerance_mape': 1e-4,
    'tolerance_prediction': 1e-6,
    'enable_detailed_comparison': True,
}


# =============================================================================
# PRODUCT-SPECIFIC DEFAULTS
# =============================================================================

PRODUCT_DEFAULTS = {
    'FlexGuard_6Y20B': {
        'product_name': 'FlexGuard_6Y20B',
        'buffer': 20,
        'term_years': 6,
        'product_type': 'rila',
    },
    'FlexGuard_6Y10B': {
        'product_name': 'FlexGuard_6Y10B',
        'buffer': 10,
        'term_years': 6,
        'product_type': 'rila',
    },
    'FlexGuard_10Y20B': {
        'product_name': 'FlexGuard_10Y20B',
        'buffer': 20,
        'term_years': 10,
        'product_type': 'rila',
    },
    'FlexGuard_1Y10B': {
        'product_name': 'FlexGuard_1Y10B',
        'buffer': 10,
        'term_years': 1,
        'product_type': 'rila',
    },
}


# =============================================================================
# UNIFIED DEFAULTS ACCESSOR
# =============================================================================

DEFAULTS = {
    'bootstrap': BOOTSTRAP_DEFAULTS,
    'cross_validation': CROSS_VALIDATION_DEFAULTS,
    'ridge': RIDGE_DEFAULTS,
    'business_filter': BUSINESS_FILTER_DEFAULTS,
    'economic_constraint': ECONOMIC_CONSTRAINT_DEFAULTS,
    'inference': INFERENCE_DEFAULTS,
    'rate_scenario': RATE_SCENARIO_DEFAULTS,
    'confidence_interval': CONFIDENCE_INTERVAL_DEFAULTS,
    'tableau_formatting': TABLEAU_FORMATTING_DEFAULTS,
    'visualization': VISUALIZATION_DEFAULTS,
    'validation_framework': VALIDATION_FRAMEWORK_DEFAULTS,
    # Feature collections
    'competitive_features': COMPETITIVE_FEATURES,
    'inference_features': INFERENCE_FEATURES,
    'model_features': MODEL_FEATURES,
    'benchmark_features': BENCHMARK_FEATURES,
    'base_features': BASE_FEATURES,
    # Product-specific
    'products': PRODUCT_DEFAULTS,
}


def get_default(
    category: str,
    key: Optional[str] = None,
    **overrides: Any
) -> Any:
    """Get default value(s) with optional overrides.

    Parameters
    ----------
    category : str
        Default category (e.g., 'bootstrap', 'inference')
    key : Optional[str], default=None
        Specific key within category. If None, returns entire category dict.
    **overrides : Any
        Key-value pairs to override defaults

    Returns
    -------
    Any
        Default value or merged dict with overrides

    Examples
    --------
    >>> n_samples = get_default('bootstrap', 'n_samples')
    1000

    >>> config = get_default('bootstrap', n_samples=500)
    {'n_samples': 500, 'random_state': 42, ...}
    """
    if category not in DEFAULTS:
        raise KeyError(f"Unknown default category: {category}. "
                       f"Available: {list(DEFAULTS.keys())}")

    defaults = DEFAULTS[category]

    if key is not None:
        if isinstance(defaults, dict):
            return overrides.get(key, defaults.get(key))
        elif isinstance(defaults, (tuple, list)):
            raise TypeError(f"Cannot access key '{key}' on sequence default")
        return defaults

    # Return full category with overrides applied
    if isinstance(defaults, dict):
        return {**defaults, **overrides}
    elif overrides:
        raise TypeError(f"Cannot apply overrides to non-dict default")
    return defaults


def get_product_defaults(product_code: str) -> Dict[str, Any]:
    """Get product-specific default configuration.

    Parameters
    ----------
    product_code : str
        Product identifier (e.g., 'FlexGuard_6Y20B')

    Returns
    -------
    Dict[str, Any]
        Product-specific defaults

    Raises
    ------
    KeyError
        If product_code not found
    """
    if product_code not in PRODUCT_DEFAULTS:
        raise KeyError(f"Unknown product: {product_code}. "
                       f"Available: {list(PRODUCT_DEFAULTS.keys())}")
    return PRODUCT_DEFAULTS[product_code].copy()


def get_feature_list(category: str) -> List[str]:
    """Get feature list by category.

    Parameters
    ----------
    category : str
        Feature category: 'competitive', 'inference', 'model', 'benchmark', 'base'

    Returns
    -------
    List[str]
        List of feature names
    """
    mapping = {
        'competitive': COMPETITIVE_FEATURES,
        'inference': INFERENCE_FEATURES,
        'model': MODEL_FEATURES,
        'benchmark': BENCHMARK_FEATURES,
        'base': BASE_FEATURES,
    }
    if category not in mapping:
        raise KeyError(f"Unknown feature category: {category}")
    return list(mapping[category])


__all__ = [
    # Default dictionaries
    "DEFAULTS",
    "BOOTSTRAP_DEFAULTS",
    "CROSS_VALIDATION_DEFAULTS",
    "RIDGE_DEFAULTS",
    "BUSINESS_FILTER_DEFAULTS",
    "ECONOMIC_CONSTRAINT_DEFAULTS",
    "INFERENCE_DEFAULTS",
    "RATE_SCENARIO_DEFAULTS",
    "CONFIDENCE_INTERVAL_DEFAULTS",
    "TABLEAU_FORMATTING_DEFAULTS",
    "VISUALIZATION_DEFAULTS",
    "VALIDATION_FRAMEWORK_DEFAULTS",
    "PRODUCT_DEFAULTS",
    # Feature collections
    "COMPETITIVE_FEATURES",
    "INFERENCE_FEATURES",
    "MODEL_FEATURES",
    "BENCHMARK_FEATURES",
    "BASE_FEATURES",
    # Accessor functions
    "get_default",
    "get_product_defaults",
    "get_feature_list",
]
